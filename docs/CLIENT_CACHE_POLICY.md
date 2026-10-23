# Checkpoint placement by client IP

`cache_tail_user_ips` is an opt-in server setting for clients that send stable
history followed by a short, changing user message, such as a bot's timestamped
reply notification:

```json
{
  "cache_tail_user_ips": ["10.0.0.2"]
}
```

For these peers, the engine saves its turn checkpoint before the final user
message. Other peers use the engine's ordinary checkpoint policy. Prompt text,
roles, generation settings, and token/image validation are unchanged. The
tokenizer supplies the user-role token; do not copy a token ID between models.

Remove a global `--tail-role-token user-token-id` from the engine arguments when
replacing a global bot policy with this setting. Requests without a peer override
still inherit any explicit engine default, including a system-tail role used by
`effort_position: "end"`.

Matching uses the actual HTTP socket peer. Forwarded/X-Forwarded-For/X-Real-IP
headers and JSON fields cannot select the policy. IPv4-mapped IPv6 addresses are
normalized; entries must be literal IP addresses, not hostnames or subnets.
Every client on a matching host receives the policy, including coding clients on
that same host. Behind a proxy, the socket peer is the proxy's address.

The policy applies to Chat Completions, Anthropic Messages and Responses, with
streaming or without it. Internal continuations retain that HTTP request's
policy. Each new GEN/GENI initializes its `tail_role` from the engine default,
so an override does not carry into the next client's request.

The IP allowlist lives in the run configuration and matching runs in Python;
no IP address is compiled into the engine. Changing the list requires restarting
the server with the updated configuration, without rebuilding the engine. The
C++ binary implements only the generic per-request checkpoint override.

The feature requires an engine advertising `request_tail_role=1`. The server
checks the binary before loading a model and refuses an unsupported configuration
instead of silently ignoring the setting. Continuous batching (`--batch` or
`--slots`) is not supported. Ordinary single-stream and layer-split execution
use the same checkpoint code.

The wire key is `tail_role=N`; `-1` disables the role-specific cut for that request.
The server's `_strata_tail_role_token` is internal and is removed from incoming
JSON before selecting a policy. It is not a public HTTP request option.

Validation: `python -m unittest serve.test_client_cache` exercises actual HTTP
handlers for all three APIs, streaming and non-streaming, exact/mapped IPs,
forged headers/body fields, disabled defaults, configuration errors, engine
capability checks and alternating clients. The Windows deployment additionally
tests real requests from localhost and `10.0.0.2`; its measurements are recorded
in the Penggu deployment log.

The final deployed settings and measurements are collected in
[the Penggu optimization summary](PENGGU_OPTIMIZATION_2026-10-23.zh-CN.md).
