"""Identity, protocol and HTTP propagation for the native context pool; no model required."""
import contextlib
import io
import json
import queue
import threading
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from serve import conversation_pool
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve


class Identity(unittest.TestCase):
    def test_parent_and_subtask_are_different(self):
        def req(thread):
            return {"prompt_cache_key": "shared", "client_metadata": {"x-codex-turn-metadata": json.dumps(
                {"session_id": "parent", "thread_id": thread, "request_kind": "turn"})}}
        a, b = conversation_pool.identity(req("parent")), conversation_pool.identity(req("worker"))
        self.assertNotEqual(a[0], b[0])
        self.assertEqual(a[0], conversation_pool.identity({"session_id": "parent", "thread_id": "parent"})[0])
        self.assertEqual(len(a[0]), 64)

    def test_missing_ids_share_anonymous(self):
        for req in ({}, {"request_id": "one"}, {"id": "two"}, {"prompt_cache_key": "group"},
                    {"metadata": {"user_id": "user-a"}}, {"session_id": 123},
                    {"conversation_id": " "}, {"conversation_id": "x" * 1025},
                    {"client_metadata": {"x-codex-turn-metadata": "{bad json"}}):
            self.assertEqual(conversation_pool.identity(req), ("", "anonymous"), req)

    def test_headers_metadata_and_explicit_id(self):
        expected = conversation_pool.identity({"conversation_id": "chat-A"})[0]
        for req in ({"strata_conversation_id": "chat-A"}, {"metadata": {"conversation_id": "chat-A"}},
                    {"_strata_pool_headers": {"x-conversation-id": "chat-A"}}):
            self.assertEqual(conversation_pool.identity(req)[0], expected)
        self.assertEqual(conversation_pool.identity({"strata_conversation_id": "chat-A", "session_id": "ignored"})[0],
                         expected)

    def test_claude_metadata_session(self):
        sid = "01234567-89ab-cdef-0123-456789abcdef"
        expected = conversation_pool.identity({"session_id": sid})[0]
        for user in (json.dumps({"account_uuid": "account", "session_id": sid}), "user_a_account_b_session_" + sid):
            self.assertEqual(conversation_pool.identity({"metadata": {"user_id": user}})[0], expected)

    def test_protocol_is_opt_in_and_ids_are_hashed(self):
        obj = SimpleNamespace(info={})
        req = {"conversation_id": "private\nGEN 100 1,2,3"}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(StrataEngine.pool_request_key(obj, req, 100), "")
        self.assertEqual(output.getvalue(), "")
        obj.info["conversation_pool_tokens"] = 524288
        with contextlib.redirect_stdout(io.StringIO()) as output:
            key = StrataEngine.pool_request_key(obj, req, 100)
        self.assertRegex(key, r"^ pool_id=[a-f0-9]{64}$")
        self.assertNotIn("private", output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(StrataEngine.pool_request_key(obj, {}, 100), " pool_id=anon")

    def test_pool_metrics(self):
        state = conversation_pool.state_line("POOL tokens=75000 budget=524288 named=1 anonymous=1 evictions=2 "
                                             "active=abc channels=abc:65000,anon:10000")
        self.assertEqual(state["used_tokens"], sum(c["tokens"] for c in state["channels"]))
        self.assertTrue(state["anonymous_used"])
        self.assertEqual(state["budget_tokens"], 524288)


class ShortInputRouting(unittest.TestCase):
    def test_long_short_short_return_preserves_named_baseline(self):
        router = conversation_pool.InputRouter(3)
        self.assertEqual(router.route("a", 90589), ("a", 0))
        self.assertEqual(router.route("a", 502), ("", 90589))
        self.assertEqual(router.route("a", 700), ("", 90589))
        self.assertEqual(router.route("a", 91535), ("a", 90589))

    def test_short_limit_and_drop_ratio_boundaries(self):
        for previous, current, anonymous in ((12000, 2999, True), (12000, 3000, False),
                                             (3000, 750, True), (3000, 751, False),
                                             (2999, 100, False), (500, 100, False)):
            with self.subTest(previous=previous, current=current):
                router = conversation_pool.InputRouter(3)
                router.route("a", previous)
                self.assertEqual(router.route("a", current), ("" if anonymous else "a", previous))

    def test_other_ids_and_anonymous_requests_do_not_replace_baseline(self):
        router = conversation_pool.InputRouter(3)
        router.route("a", 90589)
        self.assertEqual(router.route("b", 502), ("b", 0))
        self.assertEqual(router.route("", 90000), ("", 0))
        self.assertEqual(router.route("", 502), ("", 0))
        self.assertEqual(router.route("a", 502), ("", 90589))

    def test_compares_latest_named_input_instead_of_historical_maximum(self):
        router = conversation_pool.InputRouter(3)
        router.route("a", 90000)
        self.assertEqual(router.route("a", 5000), ("a", 90000))
        self.assertEqual(router.route("a", 1500), ("a", 5000))
        self.assertEqual(router.route("a", 500), ("a", 1500))

    def test_named_baselines_are_bounded_and_follow_lru(self):
        router = conversation_pool.InputRouter(2)
        router.route("a", 9000)
        router.route("b", 12000)
        router.route("a", 9500)
        router.route("c", 15000)
        self.assertEqual(router.route("a", 502), ("", 9500))
        self.assertEqual(router.route("b", 502), ("b", 0))

    def test_fallback_log_explains_route_without_raw_identity(self):
        obj = SimpleNamespace(info={"conversation_pool_tokens": 524288, "conversation_cache_slots": 4})
        req = {"conversation_id": "private-session"}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            named = StrataEngine.pool_request_key(obj, req, 90589)
            short = StrataEngine.pool_request_key(obj, req, 502)
        self.assertNotEqual(named, " pool_id=anon")
        self.assertEqual(short, " pool_id=anon")
        self.assertIn("reason=short_input", output.getvalue())
        self.assertIn("previous_input=90589", output.getvalue())
        self.assertIn("input=502", output.getvalue())
        self.assertNotIn("private-session", output.getvalue())

    def test_generate_sends_effective_channel_and_keeps_continuations_anonymous(self):
        engine = StrataEngine("unused", [], lazy=True)
        engine.info = {"conversation_pool_tokens": 524288, "conversation_cache_slots": 4}
        engine.proc = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
        engine.ended = False
        engine.lines = queue.Queue()

        def generate(count, sampling, embeddings=None, generated=0):
            engine.lines.put(f"DONE {generated} {count} 1 1 stop 0 0 0")
            with contextlib.redirect_stdout(io.StringIO()):
                list(engine.generate([7] * count, 1, sampling, threading.Event(), embeddings))
            return engine.proc.stdin.getvalue().splitlines()[-1].split()

        req = {"conversation_id": "a"}
        named = "pool_id=" + conversation_pool.identity(req)[0]
        self.assertIn(named, generate(12000, req))
        short = {**req, "_strata_pool_input_tokens": 502}
        self.assertIn("pool_id=anon", generate(502, short))
        self.assertIn("pool_id=anon", generate(4500, short, embeddings="image.bin"))
        self.assertIn(named, generate(12100, req))
        # A long output must not turn an ordinary short conversation into a long-input baseline.
        self.assertIn("pool_id=" + conversation_pool.identity({"conversation_id": "b"})[0],
                      generate(500, {"conversation_id": "b"}, generated=10000))
        self.assertNotIn("pool_id=anon", generate(400, {"conversation_id": "b"}))

    def test_service_wrapup_preserves_original_input_and_anonymous_route(self):
        from serve.test_server import ThinkingEngine

        class RoutedThinkingEngine(ThinkingEngine):
            THOUGHT = "Let me think step by step about two plus two. " * 100

            def __init__(self, tok):
                super().__init__(tok)
                self.info = {"conversation_pool_tokens": 524288, "conversation_cache_slots": 4}
                self.routes = []
                self.inputs = []

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                self.inputs.append(sampling.get("_strata_pool_input_tokens", len(ids)))
                self.routes.append(StrataEngine.pool_request_key(self, sampling, self.inputs[-1]))
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        tok = ByteTokenizer()
        engine = RoutedThinkingEngine(tok)
        svc = Service(engine, tok, ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        req = {"conversation_id": "a", "reasoning_budget_tokens": 3100, "_strata_pool_input_tokens": 90000}
        prompt = tok.encode("<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n",
                            parse_special=True)
        with contextlib.redirect_stdout(io.StringIO()):
            StrataEngine.pool_request_key(engine, req, 90589)
            list(svc.run(prompt, True, [], 4500, req, threading.Event()))
        self.assertEqual(engine.routes, [" pool_id=anon", " pool_id=anon"])
        self.assertEqual(engine.inputs, [len(prompt), len(prompt)])
        self.assertGreater(len(engine.prompts[1]), 3000)
        self.assertEqual(req["_strata_pool_input_tokens"], 90000, "leave the caller's request unchanged")

    def test_reinitializing_engine_clears_baselines_and_pool_state(self):
        engine = StrataEngine("unused", [], lazy=True)
        engine.info = {"conversation_pool_tokens": 524288}
        req = {"conversation_id": "a"}
        with contextlib.redirect_stdout(io.StringIO()):
            engine.pool_request_key(req, 90589)
        engine.pool_state = {"active": "old"}
        engine.__init__("unused", [], lazy=True)
        self.assertEqual(engine.pool_state, {})
        engine.info = {"conversation_pool_tokens": 524288}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertNotEqual(engine.pool_request_key(req, 502), " pool_id=anon")


class PoolRecordingEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.identity = conversation_pool.identity(sampling)
        self.input_tokens = sampling.get("_strata_pool_input_tokens", len(ids))
        self.prompt_tokens = len(ids)
        self.pool_key = StrataEngine.pool_request_key(self, sampling, self.input_tokens)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class HttpIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = PoolRecordingEngine(tok, "ok", max_context=20000)
        cls.engine.info = {"conversation_pool_tokens": 524288, "conversation_cache_slots": 4}
        cls.service = Service(cls.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        cls.httpd = serve(cls.service, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def setUp(self):
        self.engine.pool_input_router = None

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **(headers or {})})
        with self.opener.open(req, timeout=15) as result:
            return json.load(result)

    def test_chat_header_and_anonymous_fallback(self):
        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
        self.post("/v1/chat/completions", body, {"X-Conversation-ID": "a"})
        self.assertEqual(self.engine.identity[0], conversation_pool.identity({"conversation_id": "a"})[0])
        self.post("/v1/chat/completions", {**body, "_strata_pool_headers": {"x-conversation-id": "forged"}})
        self.assertEqual(self.engine.identity, ("", "anonymous"))

    def test_responses_metadata_reaches_engine(self):
        meta = {"x-codex-turn-metadata": json.dumps({"session_id": "root", "thread_id": "child"})}
        self.post("/v1/responses", {"model": "test", "input": "hi", "max_output_tokens": 4,
                                   "client_metadata": meta})
        self.assertEqual(self.engine.identity[0], conversation_pool.identity({"client_metadata": meta})[0])

    def test_anthropic_metadata_reaches_engine(self):
        metadata = {"user_id": json.dumps({"session_id": "claude-a"})}
        self.post("/v1/messages", {"model": "test", "messages": [{"role": "user", "content": "hi"}],
                                  "max_tokens": 4, "metadata": metadata})
        self.assertEqual(self.engine.identity[0], conversation_pool.identity({"metadata": metadata})[0])

    def test_metrics_describe_total_pool(self):
        with self.opener.open(self.base + "/metrics", timeout=10) as result:
            pool = json.load(result)["conversation_pool"]
        self.assertEqual((pool["budget_tokens"], pool["named_slots"], pool["anonymous_slots"]), (524288, 3, 1))

    def test_all_apis_divert_same_id_short_requests_and_ignore_forged_input_count(self):
        for api in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
            with self.subTest(api=api):
                self.engine.pool_input_router = None
                headers = {"X-Conversation-ID": "main"}

                def request(text):
                    body = {"model": "test", "_strata_pool_input_tokens": 90000}
                    if api == "/v1/responses":
                        body.update(input=text, max_output_tokens=4)
                    else:
                        body.update(messages=[{"role": "user", "content": text}], max_tokens=4)
                    self.post(api, body, headers)
                    self.assertEqual(self.engine.input_tokens, self.engine.prompt_tokens)
                    return self.engine.pool_key

                named = request("long context " * 600)
                self.assertGreater(self.engine.input_tokens, 3000)
                self.assertNotEqual(named, " pool_id=anon")
                self.assertEqual(request("short side task"), " pool_id=anon")
                self.assertEqual(request("another short side task"), " pool_id=anon")
                self.assertEqual(request("long context " * 610), named)


if __name__ == "__main__":
    unittest.main()
