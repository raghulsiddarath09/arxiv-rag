// load_test.js - Day 17 k6 load test
// Tests your FastAPI backend under concurrent user load
// Run with: k6 run load_test.js

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// ============================================================
// Custom metrics
// ============================================================
const cacheHitRate    = new Rate('cache_hit_rate');
const coldQueryTime   = new Trend('cold_query_latency');
const cachedQueryTime = new Trend('cached_query_latency');
const errorRate       = new Rate('error_rate');

// ============================================================
// Test configuration — 3 stages
// Stage 1: ramp up    0 → 10 users over 30s
// Stage 2: sustained  10 users for 60s
// Stage 3: ramp down  10 → 0 users over 10s
// ============================================================
export const options = {
  stages: [
    { duration: '30s', target: 10 },   // ramp up
    { duration: '60s', target: 10 },   // sustained load
    { duration: '10s', target: 0  },   // ramp down
  ],
  thresholds: {
    // 95% of all requests must complete under 10 seconds
    http_req_duration: ['p(95)<10000'],
    // Error rate must stay below 10%
    error_rate: ['rate<0.1'],
  },
};

const BASE_URL = 'http://127.0.0.1:8000';

// Questions pool — mix of repeated (cache hits) and unique (cache misses)
const REPEATED_QUESTIONS = [
  "What is BERT?",
  "What is LoRA?",
  "What is RAG?",
  "How does attention mechanism work?",
  "What is dense passage retrieval?",
];

const headers = { 'Content-Type': 'application/json' };

// ============================================================
// Setup — runs once before the test, clears cache for clean run
// ============================================================
export function setup() {
  console.log('Clearing cache before load test...');
  http.del(`${BASE_URL}/cache`);

  // Warm up cache with repeated questions
  console.log('Warming up cache with repeated questions...');
  for (const question of REPEATED_QUESTIONS) {
    http.post(
      `${BASE_URL}/query`,
      JSON.stringify({ question, session_id: 'warmup' }),
      { headers }
    );
  }
  console.log('Cache warmed. Starting load test...');
}

// ============================================================
// Default function — runs once per virtual user per iteration
// ============================================================
export default function () {
  const sessionId = `user_${__VU}`;  // __VU = virtual user number

  // 70% of requests use repeated questions (cache hits)
  // 30% use unique questions (cache misses)
  const useCache = Math.random() < 0.7;

  let question;
  if (useCache) {
    // Pick from repeated questions → should hit cache
    question = REPEATED_QUESTIONS[Math.floor(Math.random() * REPEATED_QUESTIONS.length)];
  } else {
    // Unique question → cache miss → full pipeline
    question = `What is machine learning technique number ${__VU}_${Date.now()}?`;
  }

  // ── POST /query ──────────────────────────────────────────
  const queryRes = http.post(
    `${BASE_URL}/query`,
    JSON.stringify({ question, session_id: sessionId }),
    { headers, timeout: '30s' }
  );

  // Check response
  const queryOk = check(queryRes, {
    'query status 200':     (r) => r.status === 200,
    'has answer field':     (r) => JSON.parse(r.body).answer !== undefined,
    'has cached field':     (r) => JSON.parse(r.body).cached !== undefined,
    'has sources field':    (r) => JSON.parse(r.body).sources !== undefined,
  });

  if (!queryOk) {
    errorRate.add(1);
    console.log(`ERROR: ${queryRes.status} - ${queryRes.body}`);
  } else {
    errorRate.add(0);
    const body = JSON.parse(queryRes.body);

    // Track cache hit rate
    cacheHitRate.add(body.cached ? 1 : 0);

    // Track latency separately for cached vs cold
    if (body.cached) {
      cachedQueryTime.add(queryRes.timings.duration);
    } else {
      coldQueryTime.add(queryRes.timings.duration);
    }
  }

  // ── GET /health every 5th iteration ─────────────────────
  if (__ITER % 5 === 0) {
    const healthRes = http.get(`${BASE_URL}/health`);
    check(healthRes, {
      'health status 200': (r) => r.status === 200,
      'redis connected':   (r) => JSON.parse(r.body).redis === 'connected',
    });
  }

  sleep(1);  // 1 second between requests per virtual user
}

// ============================================================
// Teardown — runs once after test completes
// ============================================================
export function teardown() {
  console.log('Load test complete.');
  const statsRes = http.get(`${BASE_URL}/cache/stats`);
  console.log(`Final cache stats: ${statsRes.body}`);
}