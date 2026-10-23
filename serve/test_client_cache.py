"""Socket-peer cache routing through all HTTP APIs, without a model or GPU."""
import json
from pathlib import Path
import tempfile
import unittest
import urllib.request
from unittest import mock

from serve.client_cache import CAPABILITY, ClientCachePolicy, TAIL_ROLE_KEY
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve


class PolicyTests(unittest.TestCase):
    def config(self, values):
        tok = mock.Mock()
        tok.encode.return_value = [846]
        return ClientCachePolicy.from_config({"cache_tail_user_ips": values}, tok)

    def test_default_is_opt_in_and_request_stays_identical(self):
        tok = mock.Mock()
        policy = ClientCachePolicy.from_config({}, tok)
        req = {"messages": [{"role": "user", "content": "unchanged"}]}
        self.assertIs(policy.request(req, "10.0.0.2"), req)
        tok.encode.assert_not_called()
        self.assertNotIn("tail_role=", StrataEngine.sampling_keys(req))

    def test_exact_ip_and_ipv4_mapped_ipv6(self):
        policy = self.config(["10.0.0.2", "2001:db8::1"])
        for peer in ("10.0.0.2", "::ffff:10.0.0.2", "2001:db8:0::1"):
            with self.subTest(peer=peer):
                req = {"messages": []}
                result = policy.request(req, peer)
                self.assertEqual(result[TAIL_ROLE_KEY], 846)
                self.assertNotIn(TAIL_ROLE_KEY, req)
        for peer in ("10.0.0.20", "10.0.0.3", "127.0.0.1", "invalid"):
            self.assertNotIn(TAIL_ROLE_KEY, policy.request({}, peer))

    def test_client_field_is_stripped_or_overridden(self):
        req = {TAIL_ROLE_KEY: 999, "messages": []}
        policy = self.config(["10.0.0.2"])
        self.assertNotIn(TAIL_ROLE_KEY, policy.request(req, "127.0.0.1"))
        self.assertNotIn(TAIL_ROLE_KEY, ClientCachePolicy().request(req, "10.0.0.2"))
        self.assertEqual(policy.request(req, "10.0.0.2")[TAIL_ROLE_KEY], 846)
        self.assertEqual(req[TAIL_ROLE_KEY], 999)

    def test_bad_configuration_is_rejected(self):
        for values in (None, "10.0.0.2", [2], ["10.0.0.0/24"], ["localhost"], ["10.0.0.256"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.config(values)
        for tokens in ([], [1, 2], [True], [-1]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                ClientCachePolicy.from_config({"cache_tail_user_ips": ["10.0.0.2"]},
                                              mock.Mock(encode=mock.Mock(return_value=tokens)))

    def test_engine_capability_required_only_when_enabled(self):
        ClientCachePolicy().check_engine("nonexistent", [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engine"
            path.write_bytes(b"old engine")
            with self.assertRaises(ValueError):
                self.config(["10.0.0.2"]).check_engine(path, [])
            path.write_bytes(CAPABILITY)
            self.config(["10.0.0.2"]).check_engine(path, [])
            for args in (["--batch", "2"], ["--slots", "2"]):
                with self.assertRaises(ValueError):
                    self.config(["10.0.0.2"]).check_engine(path, args)

    def test_wire_field_accepts_only_integer_tokens(self):
        for value in (None, True, False, "846", 846.0, -2):
            self.assertNotIn("tail_role=", StrataEngine.sampling_keys({TAIL_ROLE_KEY: value}))
        for value in (-1, 0, 846):
            self.assertIn(f"tail_role={value}", StrataEngine.sampling_keys({TAIL_ROLE_KEY: value}))


class RecordingEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.sampling = dict(sampling or {})
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingEngine(tok, "</think>\n\nOK", max_context=4096)
        cls.svc = Service(cls.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, api, stream=False, headers=None, **extra):
        body = {"model": "fixture", "stream": stream, **extra}
        if api == "responses":
            body.update(input="hello", max_output_tokens=16)
        else:
            body.update(messages=[{"role": "user", "content": "hello"}], max_tokens=16)
        req = urllib.request.Request(self.base + "/v1/" + api, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **(headers or {})})
        with self.opener.open(req, timeout=10) as response:
            self.assertEqual(response.status, 200)
            response.read()
        return StrataEngine.sampling_keys(self.engine.sampling)

    def test_matching_peer_reaches_each_api_streaming_and_nonstreaming(self):
        self.svc.client_cache_policy = ClientCachePolicy(frozenset({"127.0.0.1"}), 846)
        for api in ("chat/completions", "messages", "responses"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    self.assertIn("tail_role=846", self.post(api, stream, **{TAIL_ROLE_KEY: 999}))

    def test_other_peers_and_forged_headers_or_body_cannot_enable_it(self):
        self.svc.client_cache_policy = ClientCachePolicy(frozenset({"10.0.0.2"}), 846)
        headers = {"X-Forwarded-For": "10.0.0.2", "X-Real-IP": "10.0.0.2", "Forwarded": "for=10.0.0.2"}
        for api in ("chat/completions", "messages", "responses"):
            with self.subTest(api=api):
                self.assertNotIn("tail_role=", self.post(api, headers=headers, **{TAIL_ROLE_KEY: 846, "tail_role": 846}))

    def test_alternating_requests_do_not_leak_policy(self):
        for enabled in (True, False, True, False):
            ips = frozenset({"127.0.0.1" if enabled else "10.0.0.2"})
            self.svc.client_cache_policy = ClientCachePolicy(ips, 846)
            self.assertEqual("tail_role=846" in self.post("chat/completions"), enabled)


if __name__ == "__main__":
    unittest.main()
