// Base API URL - update this to match your FastAPI server
const API_URL = 'http://localhost:8000';

// DOM Elements - Full Processing Tab
const fullFileInput = document.getElementById('full-file-input');
const fullDropzone = document.getElementById('full-dropzone');
const fullProcessBtn = document.getElementById('full-process-btn');
const fullResults = document.getElementById('full-results');
const fullOriginalImage = document.getElementById('full-original-image');
const fullProcessedImage = document.getElementById('full-processed-image');
const fullCarViewResult = document.getElementById('full-car-view-result');
const carViewText = document.getElementById('car-view-text');
const fullLicenseResult = document.getElementById('full-license-result');
const licenseText = document.getElementById('license-text');
const fullVehicleDetails = document.getElementById('full-vehicle-details');

// DOM Elements - Classification Only Tab
const classifyFileInput = document.getElementById('classify-file-input');
const classifyDropzone = document.getElementById('classify-dropzone');
const classifyBtn = document.getElementById('classify-btn');
const classificationResults = document.getElementById('classification-results');
const classifyResultsContainer = document.getElementById('classify-results-container');

// DOM Elements - License Plate Lookup Tab
const plateInput = document.getElementById('plate-input');
const lookupBtn = document.getElementById('lookup-btn');
const lookupResults = document.getElementById('lookup-results');
const lookupPlateDisplay = document.getElementById('lookup-plate-display');

// Step status elements
const stepStatuses = {
    step1: document.getElementById('step1-status'),
    step2: document.getElementById('step2-status'),
    step3: document.getElementById('step3-status'),
    step4: document.getElementById('step4-status'),
    step5: document.getElementById('step5-status')
};

// Tab navigation
const tabs = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');

// Initialize file inputs and dropzones
let fullSelectedFile = null;
let classifySelectedFiles = [];

// Tab Navigation
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs and contents
        tabs.forEach(t => t.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        
        // Add active class to selected tab and content
        tab.classList.add('active');
        const tabId = tab.dataset.tab;
        document.getElementById(tabId).classList.add('active');
    });
});

// Dropzone functionality for full processing
fullDropzone.addEventListener('click', () => {
    fullFileInput.click();
});

fullDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    fullDropzone.classList.add('border-blue-500');
});

fullDropzone.addEventListener('dragleave', () => {
    fullDropzone.classList.remove('border-blue-500');
});

fullDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    fullDropzone.classList.remove('border-blue-500');
    
    if (e.dataTransfer.files.length > 0) {
        fullSelectedFile = e.dataTransfer.files[0];
        updateFullDropzoneText();
        fullProcessBtn.disabled = false;
    }
});

fullFileInput.addEventListener('change', () => {
    if (fullFileInput.files.length > 0) {
        fullSelectedFile = fullFileInput.files[0];
        updateFullDropzoneText();
        fullProcessBtn.disabled = false;
    }
});

// Dropzone functionality for classify only
classifyDropzone.addEventListener('click', () => {
    classifyFileInput.click();
});

classifyDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    classifyDropzone.classList.add('border-blue-500');
});

classifyDropzone.addEventListener('dragleave', () => {
    classifyDropzone.classList.remove('border-blue-500');
});

classifyDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    classifyDropzone.classList.remove('border-blue-500');
    
    if (e.dataTransfer.files.length > 0) {
        classifySelectedFiles = Array.from(e.dataTransfer.files);
        updateClassifyDropzoneText();
        classifyBtn.disabled = false;
    }
});

classifyFileInput.addEventListener('change', () => {
    if (classifyFileInput.files.length > 0) {
        classifySelectedFiles = Array.from(classifyFileInput.files);
        updateClassifyDropzoneText();
        classifyBtn.disabled = false;
    }
});

// Helper functions
function updateFullDropzoneText() {
    fullDropzone.innerHTML = `
        <input type="file" id="full-file-input" accept="image/*" class="hidden">
        <i class="fas fa-check-circle text-4xl text-green-500 mb-2"></i>
        <p class="text-lg">${fullSelectedFile.name}</p>
        <p class="text-sm text-gray-500">Click to change file</p>
    `;
    
    // Reconnect event listener to new input
    document.getElementById('full-file-input').addEventListener('change', () => {
        if (fullFileInput.files.length > 0) {
            fullSelectedFile = fullFileInput.files[0];
            updateFullDropzoneText();
            fullProcessBtn.disabled = false;
        }
    });
}

function updateClassifyDropzoneText() {
    classifyDropzone.innerHTML = `
        <input type="file" id="classify-file-input" accept="image/*" multiple class="hidden">
        <i class="fas fa-check-circle text-4xl text-green-500 mb-2"></i>
        <p class="text-lg">${classifySelectedFiles.length} file(s) selected</p>
        <p class="text-sm text-gray-500">Click to change files</p>
    `;
    
    // Reconnect event listener to new input
    document.getElementById('classify-file-input').addEventListener('change', () => {
        if (classifyFileInput.files.length > 0) {
            classifySelectedFiles = Array.from(classifyFileInput.files);
            updateClassifyDropzoneText();
            classifyBtn.disabled = false;
        }
    });
}

function updateStepStatus(step, status, text) {
    const statusElement = stepStatuses[step];
    
    // Remove all status classes
    statusElement.classList.remove('step-complete', 'step-pending', 'step-running', 'step-failed');
    
    // Add appropriate class and text
    switch (status) {
        case 'complete':
            statusElement.classList.add('step-complete');
            statusElement.textContent = '✓ Complete';
            break;
        case 'pending':
            statusElement.classList.add('step-pending');
            statusElement.textContent = '⦻ Pending';
            break;
        case 'running':
            statusElement.classList.add('step-running');
            statusElement.textContent = '⟳ Running...';
            break;
        case 'failed':
            statusElement.classList.add('step-failed');
            statusElement.textContent = '✗ Failed';
            break;
        case 'skipped':
            statusElement.classList.add('step-pending');
            statusElement.textContent = '⦻ Skipped';
            break;
        default:
            statusElement.classList.add('step-pending');
            statusElement.textContent = text || '⦻ Pending';
    }
}

function resetSteps() {
    updateStepStatus('step1', 'complete');
    updateStepStatus('step2', 'pending');
    updateStepStatus('step3', 'pending');
    updateStepStatus('step4', 'pending');
    updateStepStatus('step5', 'pending');
}

function showLoading(button) {
    const spinner = button.querySelector('.loading-spinner');
    const text = button.querySelector('span:not(.loading-spinner)');
    spinner.classList.remove('hide');
    text.classList.add('hide');
    button.disabled = true;
}

function hideLoading(button) {
    const spinner = button.querySelector('.loading-spinner');
    const text = button.querySelector('span:not(.loading-spinner)');
    spinner.classList.add('hide');
    text.classList.remove('hide');
    button.disabled = false;
}

function displayVehicleDetails(vehicleData, prefix = '') {
    // Vehicle Information
    if (vehicleData.maker_model || (vehicleData.manufacturer && vehicleData.vehicle_model)) {
        const makeModel = vehicleData.maker_model || `${vehicleData.manufacturer || ''} ${vehicleData.vehicle_model || ''}`;
        document.getElementById(`${prefix}make-model`).innerHTML = `<strong>Make & Model:</strong> ${makeModel}`;
    }
    
    document.getElementById(`${prefix}owner-name`).innerHTML = `<strong>Owner Name:</strong> ${vehicleData.owner_name || 'N/A'}`;
    document.getElementById(`${prefix}registration-no`).innerHTML = `<strong>Registration No:</strong> ${vehicleData.registration_no || 'N/A'}`;
    document.getElementById(`${prefix}registration-date`).innerHTML = `<strong>Registration Date:</strong> ${vehicleData.registration_date || 'N/A'}`;
    document.getElementById(`${prefix}vehicle-class`).innerHTML = `<strong>Vehicle Class:</strong> ${vehicleData.vehicle_class || 'N/A'}`;
    document.getElementById(`${prefix}fuel-type`).innerHTML = `<strong>Fuel Type:</strong> ${vehicleData.fuel_type || 'N/A'}`;
    
    if (vehicleData.vehicle_type) {
        const vehicleType = vehicleData.vehicle_type.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
        document.getElementById(`${prefix}vehicle-type`).innerHTML = `<strong>Vehicle Type:</strong> ${vehicleType}`;
    }
    
    if (vehicleData.seat_capacity) {
        document.getElementById(`${prefix}seat-capacity`).innerHTML = `<strong>Seat Capacity:</strong> ${vehicleData.seat_capacity}`;
    }
    
    if (vehicleData.vehicle_color) {
        document.getElementById(`${prefix}vehicle-color`).innerHTML = `<strong>Vehicle Color:</strong> ${vehicleData.vehicle_color}`;
    }
    
    if (vehicleData.manufacture_month_year) {
        document.getElementById(`${prefix}manufacture-year`).innerHTML = `<strong>Manufacture Year:</strong> ${vehicleData.manufacture_month_year}`;
    }
    
    if (vehicleData.rc_status) {
        document.getElementById(`${prefix}rc-status`).innerHTML = `<strong>RC Status:</strong> ${vehicleData.rc_status}`;
    }
    
    if (vehicleData.ownership || vehicleData.ownership_desc) {
        const ownership = vehicleData.ownership_desc || `${vehicleData.ownership || ''} OWNER`;
        document.getElementById(`${prefix}ownership`).innerHTML = `<strong>Ownership:</strong> ${ownership}`;
    }
    
    // Additional Details
    document.getElementById(`${prefix}rto`).innerHTML = `<strong>RTO:</strong> ${vehicleData.registration_authority || 'N/A'}`;
    
    if (vehicleData.rto_address) {
        document.getElementById(`${prefix}rto-address`).innerHTML = `<strong>RTO Address:</strong> ${vehicleData.rto_address}`;
    }
    
    if (vehicleData.state) {
        document.getElementById(`${prefix}state`).innerHTML = `<strong>State:</strong> ${vehicleData.state}`;
    }
    
    if (vehicleData.rto_phone) {
        document.getElementById(`${prefix}rto-phone`).innerHTML = `<strong>RTO Phone:</strong> ${vehicleData.rto_phone}`;
    }
    
    document.getElementById(`${prefix}engine-no`).innerHTML = `<strong>Engine No:</strong> ${vehicleData.engine_no || 'N/A'}`;
    document.getElementById(`${prefix}chassis-no`).innerHTML = `<strong>Chassis No:</strong> ${vehicleData.chassis_no || 'N/A'}`;
    document.getElementById(`${prefix}insurance-company`).innerHTML = `<strong>Insurance Company:</strong> ${vehicleData.insurance_company || 'N/A'}`;
    document.getElementById(`${prefix}insurance-valid`).innerHTML = `<strong>Insurance Valid Until:</strong> ${vehicleData.insurance_upto || 'N/A'}`;
    
    if (vehicleData.financier_name) {
        document.getElementById(`${prefix}financier`).innerHTML = `<strong>Financier:</strong> ${vehicleData.financier_name}`;
    }
    
    if (vehicleData.fitness_upto) {
        document.getElementById(`${prefix}fitness-valid`).innerHTML = `<strong>Fitness Valid Until:</strong> ${vehicleData.fitness_upto}`;
    }
    
    if (vehicleData.puc_upto) {
        document.getElementById(`${prefix}puc-valid`).innerHTML = `<strong>PUC Valid Until:</strong> ${vehicleData.puc_upto}`;
    }
    
    if (vehicleData.road_tax_paid_upto) {
        document.getElementById(`${prefix}road-tax`).innerHTML = `<strong>Road Tax Paid Until:</strong> ${vehicleData.road_tax_paid_upto}`;
    }
    
    if (vehicleData.website) {
        document.getElementById(`${prefix}website`).innerHTML = `<strong>Website:</strong> ${vehicleData.website}`;
    }
}

// Event Handlers

// Full Processing
fullProcessBtn.addEventListener('click', async () => {
    if (!fullSelectedFile) return;
    
    // Reset UI
    resetSteps();
    fullCarViewResult.classList.add('hide');
    fullLicenseResult.classList.add('hide');
    fullVehicleDetails.classList.add('hide');
    fullResults.classList.remove('hide');
    
    // Display original image
    const reader = new FileReader();
    reader.onload = (e) => {
        fullOriginalImage.src = e.target.result;
    };
    reader.readAsDataURL(fullSelectedFile);
    
    // Show loading
    showLoading(fullProcessBtn);
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', fullSelectedFile);
        
        // Get selected lookup method
        const lookupMethod = document.querySelector('input[name="lookup-method"]:checked').value;
        formData.append('lookup_method', lookupMethod);
        
        updateStepStatus('step2', 'running');
        
        // Send request to API
        const response = await axios.post(`${API_URL}/process-car-image/`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
        
        // Handle response
        if (response.data.success) {
            // Update car view
            updateStepStatus('step2', 'complete');
            carViewText.textContent = response.data.car_view.toUpperCase();
            fullCarViewResult.classList.remove('hide');
            
            // Update license plate detection
            if (response.data.license_detected) {
                updateStepStatus('step3', 'complete');
                
                // Display processed image
                fullProcessedImage.src = `data:image/jpeg;base64,${response.data.processed_image_base64}`;
                
                // Update license plate processing
                if (response.data.corrected_plate) {
                    updateStepStatus('step4', 'complete');
                    licenseText.textContent = response.data.corrected_plate;
                    fullLicenseResult.classList.remove('hide');
                    
                    // Update vehicle details
                    if (response.data.vehicle_details_found) {
                        updateStepStatus('step5', 'complete');
                        displayVehicleDetails(response.data.vehicle_details);
                        fullVehicleDetails.classList.remove('hide');
                    } else {
                        updateStepStatus('step5', 'failed');
                    }
                } else {
                    updateStepStatus('step4', 'failed');
                    licenseText.textContent = response.data.raw_plate + " (Unprocessed)";
                    fullLicenseResult.classList.remove('hide');
                    updateStepStatus('step5', 'skipped');
                }
            } else {
                updateStepStatus('step3', 'failed');
                updateStepStatus('step4', 'skipped');
                updateStepStatus('step5', 'skipped');
                
                // Display processed image anyway
                fullProcessedImage.src = `data:image/jpeg;base64,${response.data.processed_image_base64}`;
            }
        } else {
            alert('Processing failed. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing the image. Please try again.');
    } finally {
        hideLoading(fullProcessBtn);
    }
});

// Classification Only
classifyBtn.addEventListener('click', async () => {
    if (classifySelectedFiles.length === 0) return;
    
    classifyResultsContainer.innerHTML = '';
    classificationResults.classList.remove('hide');
    
    showLoading(classifyBtn);
    
    try {
        // Process each file
        for (const file of classifySelectedFiles) {
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await axios.post(`${API_URL}/classify-car/`, formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data'
                    }
                });
                
                if (response.data.success) {
                    // Create result card
                    const resultCard = document.createElement('div');
                    resultCard.className = 'bg-white rounded-lg shadow overflow-hidden';
                    
                    resultCard.innerHTML = `
                        <img src="data:image/jpeg;base64,${response.data.image_base64}" class="w-full h-48 object-cover">
                        <div class="p-4">
                            <h4 class="font-semibold">${file.name}</h4>
                            <div class="mt-2 bg-blue-50 p-2 rounded-md">
                                <p><strong>View:</strong> ${response.data.car_view.toUpperCase()}</p>
                            </div>
                        </div>
                    `;
                    
                    classifyResultsContainer.appendChild(resultCard);
                }
            } catch (error) {
                console.error(`Error processing ${file.name}:`, error);
                
                // Create error card
                const errorCard = document.createElement('div');
                errorCard.className = 'bg-white rounded-lg shadow overflow-hidden';
                
                errorCard.innerHTML = `
                    <div class="p-4">
                        <h4 class="font-semibold">${file.name}</h4>
                        <div class="mt-2 bg-red-50 p-2 rounded-md">
                            <p><strong>Error:</strong> Failed to process image</p>
                        </div>
                    </div>
                `;
                
                classifyResultsContainer.appendChild(errorCard);
            }
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while classifying images. Please try again.');
    } finally {
        hideLoading(classifyBtn);
    }
});

// License Plate Lookup
lookupBtn.addEventListener('click', async () => {
    const plate = plateInput.value.trim().toUpperCase();
    
    if (!plate) {
        alert('Please enter a license plate number.');
        return;
    }
    
    showLoading(lookupBtn);
    lookupResults.classList.add('hide');
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('plate_number', plate);
        
        // Get selected lookup method
        const lookupMethod = document.querySelector('input[name="direct-lookup-method"]:checked').value;
        formData.append('lookup_method', lookupMethod);
        
        // Send request to API
        const response = await axios.post(`${API_URL}/lookup-vehicle/`, formData);
        
        if (response.data.success) {
            // Display vehicle details
            lookupPlateDisplay.textContent = plate;
            displayVehicleDetails(response.data.vehicle_details, 'lookup-');
            lookupResults.classList.remove('hide');
        } else {
            alert('Vehicle details not found. Please check the license plate number and try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while looking up the vehicle. Please try again.');
    } finally {
        hideLoading(lookupBtn);
    }
});

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    console.log('App initialized');
});