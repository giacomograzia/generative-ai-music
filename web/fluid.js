// Shared motion helpers for the web pages (no libraries, works offline).
// Follows the apple-design skill: critically damped springs (damping 1.0, no bounce), interruptible
// (a new target re-aims the motion from the current value and velocity), and reduced motion respected.

const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)");
const _springs = new WeakMap();

/**
 * Move a value towards `target` with a critically damped spring and call apply(value) every frame.
 * el + key identify the value, so calling again before it settles re-targets smoothly from where it is.
 * response: seconds, how quickly it gets there (Apple's "response"; not a fixed duration).
 */
function springTo(el, key, target, apply, response = 0.35) {
  let map = _springs.get(el);
  if (!map) _springs.set(el, map = {});
  let s = map[key];
  if (!s || REDUCED_MOTION.matches) {          // first appearance, or reduced motion: no movement, just the value
    map[key] = { x: target, v: 0, target, running: false };
    apply(target);
    return;
  }
  s.target = target; s.apply = apply; s.omega = 2 * Math.PI / response;
  if (s.running) return;                       // already moving: it picks up the new target next frame
  s.running = true;
  let last = performance.now();
  const frame = now => {
    const dt = Math.min((now - last) / 1000, 1 / 30); last = now;
    const w = s.omega, a = -2 * w * s.v - w * w * (s.x - s.target);   // damping ratio 1.0
    s.v += a * dt; s.x += s.v * dt;
    if (Math.abs(s.x - s.target) < 5e-4 && Math.abs(s.v) < 5e-3) { s.x = s.target; s.v = 0; s.running = false; }
    s.apply(s.x);
    if (s.running) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

/** Scale an element horizontally (0..1) with a spring; compositor-friendly (transform only). */
function springScaleX(el, fraction, response) {
  springTo(el, "scaleX", Math.max(0, Math.min(1, fraction)), v => { el.style.transform = `scaleX(${v})`; }, response);
}
