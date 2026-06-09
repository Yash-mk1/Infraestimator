/* scanner/static/scanner/js/result.js */
 
(function () {
  // Image tab switcher
  document.querySelectorAll('.img-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.img-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.img-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      const target = document.getElementById('panel-' + tab.dataset.target);
      if (target) target.classList.add('active');
    });
  });
 
  // Animate subscore bars on load
  window.addEventListener('load', () => {
    document.querySelectorAll('.subscore-fill').forEach(el => {
      const w = el.style.width;
      el.style.width = '0%';
      requestAnimationFrame(() => {
        requestAnimationFrame(() => { el.style.width = w; });
      });
    });
  });
})();