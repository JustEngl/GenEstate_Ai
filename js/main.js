// State management
const state = {
    style: 'modern',
    size: 'medium',
    roof_color: 'dark',
    extra: 'none'
};

// Option buttons (Style, Size)
document.querySelectorAll('.option-button').forEach(button => {
    button.addEventListener('click', function() {
        const group = this.dataset.group;
        const value = this.dataset.value;
        document.querySelectorAll(`[data-group="${group}"]`).forEach(btn => {
            btn.classList.remove('active');
        });
        this.classList.add('active');
        state[group] = value;
    });
});

// Color swatches (Roof Color)
document.querySelectorAll('.color-swatch').forEach(swatch => {
    swatch.addEventListener('click', function() {
        const value = this.dataset.value;
        document.querySelectorAll(`[data-group="roof"]`).forEach(sw => {
            sw.classList.remove('active');
        });
        this.classList.add('active');
        state.roof_color = value;
    });
});

// Radio buttons (Extras)
document.querySelectorAll('input[name="extra"]').forEach(radio => {
    radio.addEventListener('change', function() {
        state.extra = this.value;
    });
});

// Estimate Price
async function estimatePrice() {
    const button = document.querySelector('.estimate-button');
    const priceContent = document.getElementById('priceContent');
    
    button.disabled = true;
    button.textContent = 'Calculating...';
    priceContent.innerHTML = '<div class="price-loading">Calculating...</div>';
    
    try {
        const response = await fetch('/estimate-price', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(state)
        });
        
        if (response.ok) {
            const data = await response.json();
            priceContent.innerHTML = `
                <div class="price-content">
                    <div class="price-label">Price Estimate</div>
                    <div class="price-amount">${data.formatted}</div>
                    <div class="price-subtext">Estimated market value</div>
                </div>
            `;
        } else {
            priceContent.innerHTML = '<div class="price-placeholder" style="color: #B8453C;">Error</div>';
        }
    } catch (error) {
        priceContent.innerHTML = '<div class="price-placeholder" style="color: #B8453C;">Error: ' + error.message + '</div>';
    } finally {
        button.disabled = false;
        button.textContent = 'Estimate Price';
    }
}

// Generate House Image
async function generateHouse() {
    const button = document.querySelector('.generate-button');
    const previewBox = document.getElementById('previewBox');
    
    button.disabled = true;
    button.textContent = 'Generating...';
    previewBox.innerHTML = '<span class="progress-text">Generating image...</span>';
    
    console.log('Generating with:', JSON.stringify(state));
    
    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(state)
        });
        
        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);
        
        if (response.ok) {
            const blob = await response.blob();
            console.log('Received blob:', blob.size, 'bytes, type:', blob.type);
            
            const imageUrl = URL.createObjectURL(blob);
            const img = document.createElement('img');
            img.src = imageUrl;
            img.alt = 'Generated House';
            img.onload = () => {
                console.log('Image loaded successfully');
                previewBox.innerHTML = '';
                previewBox.appendChild(img);
            };
            img.onerror = (e) => {
                console.error('Image load error:', e);
                previewBox.innerHTML = '<span class="progress-text" style="color: #B8453C;">Error loading image</span>';
            };
        } else {
            const errorText = await response.text();
            console.error('Server error:', errorText);
            previewBox.innerHTML = '<span class="progress-text" style="color: #B8453C;">Server error</span>';
        }
    } catch (error) {
        console.error('Fetch error:', error);
        previewBox.innerHTML = '<span class="progress-text" style="color: #B8453C;">Network error: ' + error.message + '</span>';
    } finally {
        button.disabled = false;
        button.textContent = 'Generate House';
    }
}