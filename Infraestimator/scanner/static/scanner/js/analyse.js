/* scanner/static/scanner/js/analyse.js */
 
(function () {
  const cameraBox    = document.getElementById('cameraBox');
  const uploadBox    = document.getElementById('uploadBox');
  const fileInput    = document.getElementById('fileInput');
  const cameraModal  = document.getElementById('cameraModal');
  const closeCamera  = document.getElementById('closeCamera');
  const cameraFeed   = document.getElementById('cameraFeed');
  const captureBtn   = document.getElementById('captureBtn');
  const snapCanvas   = document.getElementById('snapCanvas');
  const loadingEl    = document.getElementById('loadingOverlay');
  const scanForm     = document.getElementById('scanForm');
  const materialSel  = document.getElementById('materialSelect');
  const materialHid  = document.getElementById('materialHidden');
  const imageB64     = document.getElementById('imageB64');
  const imageFileHid = document.getElementById('imageFileHidden');
 
  let mediaStream = null;
 
  // ── Camera flow ─────────────────────────────────────────
  cameraBox.addEventListener('click', openCamera);
  cameraBox.addEventListener('keydown', e => { if (e.key === 'Enter') openCamera(); });
 
  function openCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      showToast('⚠ Camera not supported in this browser.');
      return;
    }
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false })
      .then(stream => {
        mediaStream = stream;
        cameraFeed.srcObject = stream;
        cameraModal.classList.add('open');
      })
      .catch(err => {
        showToast('⚠ Camera access denied: ' + err.message);
      });
  }
 
  closeCamera.addEventListener('click', stopCamera);
 
  function stopCamera() {
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }
    cameraFeed.srcObject = null;
    cameraModal.classList.remove('open');
  }
 
  captureBtn.addEventListener('click', () => {
    const vw = cameraFeed.videoWidth;
    const vh = cameraFeed.videoHeight;
    snapCanvas.width  = vw;
    snapCanvas.height = vh;
    snapCanvas.getContext('2d').drawImage(cameraFeed, 0, 0, vw, vh);
    const dataUrl = snapCanvas.toDataURL('image/jpeg', 0.92);
 
    stopCamera();
 
    // Submit via hidden form with base64
    materialHid.value = materialSel.value;
    imageB64.value    = dataUrl;
    imageFileHid.value = '';      // clear file field
 
    loadingEl.classList.add('open');
    // Remove the file input from form so only b64 is sent
    const fileField = scanForm.querySelector('input[name="image_file"]');
    if (fileField) fileField.disabled = true;
 
    scanForm.submit();
  });
 
  // ── Upload flow ─────────────────────────────────────────
  uploadBox.addEventListener('click',   () => fileInput.click());
  uploadBox.addEventListener('keydown', e => { if (e.key === 'Enter') fileInput.click(); });
 
  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (!file) return;
 
    const allowedTypes = ['image/jpeg','image/png','image/webp','image/bmp'];
    if (!allowedTypes.includes(file.type)) {
      showToast('⚠ Unsupported format. Use JPG, PNG, WEBP or BMP.');
      return;
    }
 
    if (file.size > 20 * 1024 * 1024) {
      showToast('⚠ File too large. Maximum 20 MB.');
      return;
    }
 
    showToast(`✓ ${file.name}  (${(file.size/1024/1024).toFixed(1)} MB) — submitting…`);
 
    // Transfer file to the form's hidden file input
    const dt = new DataTransfer();
    dt.items.add(file);
    imageFileHid.files = dt.files;
 
    materialHid.value = materialSel.value;
    imageB64.value    = '';     // clear b64 field
 
    // Disable b64 field so only file is sent
    imageB64.disabled = true;
 
    loadingEl.classList.add('open');
 
    setTimeout(() => scanForm.submit(), 300);   // let toast show briefly
  });
 
})();