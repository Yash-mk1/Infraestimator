/* scanner/static/scanner/js/carousel.js */
 
(function () {
  const SLIDE_COUNT = 4;
  let current = 0;
  let timer;
 
  const track = document.getElementById('carTrack');
  const dotsEl = document.getElementById('carDots');
  if (!track || !dotsEl) return;
 
  // Build dots
  for (let i = 0; i < SLIDE_COUNT; i++) {
    const d = document.createElement('div');
    d.className = 'car-dot' + (i === 0 ? ' active' : '');
    d.addEventListener('click', () => goSlide(i));
    dotsEl.appendChild(d);
  }
 
  function goSlide(index) {
    current = ((index % SLIDE_COUNT) + SLIDE_COUNT) % SLIDE_COUNT;
    track.style.transform = `translateX(-${current * 100}%)`;
    dotsEl.querySelectorAll('.car-dot').forEach((d, i) =>
      d.classList.toggle('active', i === current)
    );
    resetAuto();
  }
 
  function resetAuto() {
    clearInterval(timer);
    timer = setInterval(() => goSlide(current + 1), 4200);
  }
 
  document.getElementById('carPrev')?.addEventListener('click', () => goSlide(current - 1));
  document.getElementById('carNext')?.addEventListener('click', () => goSlide(current + 1));
 
  // Touch swipe
  let tx = 0;
  track.addEventListener('touchstart', e => { tx = e.touches[0].clientX; }, { passive: true });
  track.addEventListener('touchend',   e => {
    const dx = e.changedTouches[0].clientX - tx;
    if (Math.abs(dx) > 40) goSlide(dx < 0 ? current + 1 : current - 1);
  });
 
  resetAuto();
})();