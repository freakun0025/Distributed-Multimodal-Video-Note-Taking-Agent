-- Atomic token bucket with reservation (ARCHITECTURE §9).
--
-- KEYS[1]  bucket hash: tokens (may go negative = booked debt), ts (ms, Redis TIME)
-- ARGV[1]  refill rate, tokens per second
-- ARGV[2]  capacity (the allowed burst)
-- ARGV[3]  cost of this call
-- ARGV[4]  longest wait the caller accepts, ms
--
-- Returns 0 (go now), N > 0 (booked: go in N ms) or -N (wait of N ms exceeds ARGV[4]; nothing
-- booked). Redis TIME keeps worker clocks out of it. Booking a slot instead of re-polling serves
-- waiting workers in arrival order and still allows at most capacity + rate * window calls.

local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local max_wait_ms = tonumber(ARGV[4])

local time = redis.call("TIME")
local now = tonumber(time[1]) * 1000 + math.floor(tonumber(time[2]) / 1000)

local state = redis.call("HMGET", KEYS[1], "tokens", "ts")
local tokens = tonumber(state[1]) or capacity
local ts = tonumber(state[2]) or now
tokens = math.min(capacity, tokens + math.max(0, now - ts) * rate / 1000)

local after = tokens - cost
local wait_ms = 0
if after < 0 then
  wait_ms = math.ceil(-after * 1000 / rate)
  if wait_ms > max_wait_ms then
    return -wait_ms
  end
end

redis.call("HSET", KEYS[1], "tokens", after, "ts", now)
-- An idle bucket expires once it would be full again; a missing key reads as full.
redis.call("PEXPIRE", KEYS[1], math.ceil((capacity - after) * 1000 / rate) + 1000)
return wait_ms
