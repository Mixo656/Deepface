const srcImageInput = document.getElementById('src-image');
const tgtVideoInput = document.getElementById('tgt-video');
const imagePreviewArea = document.getElementById('image-preview');
const videoPreviewArea = document.getElementById('video-preview');
const processBtn = document.getElementById('process-btn');

let sourceImageFile = null;
let targetVideoFile = null;

// Handle Image Selection
srcImageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        sourceImageFile = file;
        const reader = new FileReader();
        reader.onload = function(event) {
            imagePreviewArea.innerHTML = `<img src="${event.target.result}" alt="Source Character">`;
            imagePreviewArea.style.display = 'block';
            checkReady();
        };
        reader.readAsDataURL(file);
    }
});

// Handle Video Selection
tgtVideoInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        targetVideoFile = file;
        const videoURL = URL.createObjectURL(file);
        videoPreviewArea.innerHTML = `<video src="${videoURL}" controls muted></video>`;
        videoPreviewArea.style.display = 'block';
        checkReady();
    }
});

function checkReady() {
    if (sourceImageFile && targetVideoFile) {
        processBtn.disabled = false;
    }
}

// Fidelity slider
const fidelitySlider = document.getElementById('fidelity-slider');
const fidelityVal = document.getElementById('fidelity-val');

if (fidelitySlider) {
    fidelitySlider.addEventListener('input', () => {
        fidelityVal.textContent = fidelitySlider.value;
    });
}

// Handle Process Button
processBtn.addEventListener('click', async () => {
    if (!sourceImageFile || !targetVideoFile) return;

    // UI Updates
    processBtn.disabled = true;
    processBtn.textContent = 'Processing...';
    document.getElementById('status-section').style.display = 'block';
    document.getElementById('result-section').style.display = 'none';

    // Form Data
    const formData = new FormData();
    formData.append('source_image', sourceImageFile);
    formData.append('target_video', targetVideoFile);
    formData.append('enhance', document.getElementById('enhance-toggle').checked);
    formData.append('restore_type', document.getElementById('restore-type').value);
    formData.append('fidelity', document.getElementById('fidelity-slider').value / 100);

    try {
        const response = await fetch('/api/swap-video', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        pollStatus(data.job_id);

    } catch (error) {
        console.error("Error creating job:", error);
        document.getElementById('status-text').textContent = 'Error starting job';
        processBtn.disabled = false;
        processBtn.textContent = 'Generate Video';
    }
});

async function pollStatus(jobId) {
    const statusText = document.getElementById('status-text');
    const progressBar = document.getElementById('progress-bar');

    const interval = setInterval(async () => {
        try {
            const res = await fetch(`/api/status/${jobId}`);
            const data = await res.json();

            if (data.status === 'queued') {
                statusText.textContent = 'Queued... waiting for model to load.';
            } else if (data.status === 'processing') {
                statusText.textContent = `Processing Video... ${data.progress}%`;
                progressBar.style.width = `${data.progress}%`;
            } else if (data.status === 'completed') {
                clearInterval(interval);
                statusText.textContent = 'Completed!';
                progressBar.style.width = '100%';
                
                // Show result
                showResult(data.result_url);
            } else if (data.status === 'error') {
                clearInterval(interval);
                statusText.textContent = `Error: ${data.error}`;
                statusText.style.color = '#ff3366';
                processBtn.disabled = false;
                processBtn.textContent = 'Generate Video';
            }

        } catch (error) {
            console.error("Error polling status:", error);
        }
    }, 1000);
}

function showResult(url) {
    const resultSection = document.getElementById('result-section');
    const resultVideo = document.getElementById('result-video');
    const downloadBtn = document.getElementById('download-link');

    resultVideo.src = url;
    resultVideo.load();
    downloadBtn.href = url;
    
    document.getElementById('status-section').style.display = 'none';
    resultSection.style.display = 'block';
    
    // Reset Process Button
    processBtn.disabled = false;
    processBtn.textContent = 'Generate Another';
}
