// Global state management
let currentUser = null;
let predictionHistory = [];
let backendAvailable = false;
let backendUsingMock = true;

// Page management
function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // Show selected page
    document.getElementById(pageId).classList.add('active');
}

// Modal management
function showModal(modalId) {
    document.getElementById(modalId).style.display = 'block';
    document.body.style.overflow = 'hidden';
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
    document.body.style.overflow = 'auto';
}

function showLogin() {
    showModal('login-modal');
}

function showSignup() {
    showModal('signup-modal');
}

function switchToSignup() {
    closeModal('login-modal');
    showModal('signup-modal');
}

function switchToLogin() {
    closeModal('signup-modal');
    showModal('login-modal');
}

// Close modal when clicking outside
window.onclick = function(event) {
    if (event.target.classList.contains('modal')) {
        event.target.style.display = 'none';
        document.body.style.overflow = 'auto';
    }
}

// Authentication handlers
function handleLogin(event) {
    event.preventDefault();
    
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;
    
    // Simulate login process
    showLoadingState('Signing you in...');
    
    setTimeout(() => {
        // Mock successful login
        currentUser = {
            id: 1,
            name: 'John Doe',
            email: email,
            company: 'Diamond Experts Inc.',
            plan: 'Professional',
            predictionsUsed: 1247,
            predictionsLimit: 10000
        };
        
        // Close modal and show dashboard
        closeModal('login-modal');
        showPage('dashboard-page');
        updateUserInterface();
        hideLoadingState();
        
        // Show success message
        showNotification('Welcome back! You have been successfully signed in.', 'success');
    }, 1500);
}

function handleSignup(event) {
    event.preventDefault();
    
    const firstName = document.getElementById('signup-firstname').value;
    const lastName = document.getElementById('signup-lastname').value;
    const email = document.getElementById('signup-email').value;
    const company = document.getElementById('signup-company').value;
    const password = document.getElementById('signup-password').value;
    
    // Simulate signup process
    showLoadingState('Creating your account...');
    
    setTimeout(() => {
        // Mock successful signup
        currentUser = {
            id: 2,
            name: `${firstName} ${lastName}`,
            email: email,
            company: company || 'Individual',
            plan: 'Starter',
            predictionsUsed: 0,
            predictionsLimit: 1000
        };
        
        // Close modal and show dashboard
        closeModal('signup-modal');
        showPage('dashboard-page');
        updateUserInterface();
        hideLoadingState();
        
        // Show welcome message
        showNotification('Welcome to DiamondAI! Your account has been created successfully.', 'success');
    }, 2000);
}

function logout() {
    currentUser = null;
    predictionHistory = [];
    showPage('landing-page');
    showNotification('You have been successfully logged out.', 'info');
}

// Dashboard section management
function showDashboardSection(sectionId) {
    // Update navigation
    document.querySelectorAll('.dashboard-nav .nav-link').forEach(link => {
        link.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Show section
    document.querySelectorAll('.dashboard-section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(sectionId + '-section').classList.add('active');
}

// Update user interface with current user data
function updateUserInterface() {
    if (currentUser) {
        document.querySelector('.user-name').textContent = currentUser.name;
        // Update other user-specific elements as needed
    }
}

// Diamond prediction handler
function handlePrediction(event) {
    event.preventDefault();
    
    // Get form data
    const formData = {
        carat: parseFloat(document.getElementById('carat').value),
        cut: document.getElementById('cut').value,
        color: document.getElementById('color').value,
        clarity: document.getElementById('clarity').value,
        depth: parseFloat(document.getElementById('depth').value),
        table: parseFloat(document.getElementById('table').value),
        x: parseFloat(document.getElementById('x').value),
        y: parseFloat(document.getElementById('y').value),
        z: parseFloat(document.getElementById('z').value)
    };
    
    // Validate form data
    if (!validatePredictionForm(formData)) {
        showNotification('Please fill in all required fields correctly.', 'error');
        return;
    }
    
    // Show loading state
    document.getElementById('prediction-placeholder').style.display = 'none';
    document.getElementById('prediction-result').style.display = 'none';
    document.getElementById('prediction-loading').style.display = 'block';

    // Call backend API
    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
    })
    .then(res => res.json())
    .then(res => {
        if (!res.success) throw new Error(res.error || 'Prediction failed');
        const p = res.prediction;
        // Normalize keys to camelCase for UI
        const prediction = {
            price: p.price,
            confidence: p.confidence,
            priceRange: p.price_range || p.priceRange,
            pricePerCarat: p.price_per_carat || p.pricePerCarat,
            category: p.category,
            categoryIcon: p.category_icon || p.categoryIcon,
            model: p.model || 'Predictor'
        };

        displayPredictionResult(prediction, formData);

        // Add to history
        predictionHistory.unshift({
            ...formData,
            ...prediction,
            timestamp: new Date().toISOString()
        });

        // Update user's prediction count
        if (currentUser) {
            currentUser.predictionsUsed++;
        }
    })
    .catch(err => {
        document.getElementById('prediction-loading').style.display = 'none';
        document.getElementById('prediction-result').style.display = 'none';
        document.getElementById('prediction-placeholder').style.display = 'block';
        showNotification(`Prediction error: ${err.message}`, 'error');
    });
}

// Validate prediction form
function validatePredictionForm(data) {
    return data.carat > 0 && 
           data.cut && 
           data.color && 
           data.clarity && 
           data.depth > 0 && 
           data.table > 0 && 
           data.x > 0 && 
           data.y > 0 && 
           data.z > 0;
}

// Calculate diamond price (mock ML prediction)
function calculateDiamondPrice(data) {
    // Mock ML calculation based on diamond characteristics
    let basePrice = 3000;
    
    // Carat weight (most important factor)
    basePrice *= Math.pow(data.carat, 2.5);
    
    // Cut quality multiplier
    const cutMultipliers = {
        'Fair': 0.8,
        'Good': 0.9,
        'Very Good': 1.0,
        'Premium': 1.1,
        'Ideal': 1.2
    };
    basePrice *= cutMultipliers[data.cut] || 1.0;
    
    // Color grade multiplier
    const colorMultipliers = {
        'D': 1.3, 'E': 1.2, 'F': 1.1, 'G': 1.0,
        'H': 0.9, 'I': 0.8, 'J': 0.7
    };
    basePrice *= colorMultipliers[data.color] || 1.0;
    
    // Clarity multiplier
    const clarityMultipliers = {
        'FL': 2.0, 'IF': 1.8, 'VVS1': 1.6, 'VVS2': 1.4,
        'VS1': 1.2, 'VS2': 1.0, 'SI1': 0.8, 'SI2': 0.6, 'I1': 0.4
    };
    basePrice *= clarityMultipliers[data.clarity] || 1.0;
    
    // Add some randomness for realism
    const variance = 0.1;
    const randomFactor = 1 + (Math.random() - 0.5) * variance;
    basePrice *= randomFactor;
    
    // Ensure minimum price
    basePrice = Math.max(basePrice, 300);
    
    const predictedPrice = Math.round(basePrice);
    const confidence = Math.round(90 + Math.random() * 8); // 90-98% confidence
    const priceRange = {
        min: Math.round(predictedPrice * 0.85),
        max: Math.round(predictedPrice * 1.15)
    };
    
    // Determine price category
    let category, categoryIcon;
    if (predictedPrice < 2000) {
        category = 'Budget-Friendly';
        categoryIcon = '💚';
    } else if (predictedPrice < 5000) {
        category = 'Mid-Range';
        categoryIcon = '💛';
    } else if (predictedPrice < 10000) {
        category = 'Premium';
        categoryIcon = '🧡';
    } else {
        category = 'Luxury';
        categoryIcon = '❤️';
    }
    
    return {
        price: predictedPrice,
        confidence: confidence,
        priceRange: priceRange,
        pricePerCarat: Math.round(predictedPrice / data.carat),
        category: category,
        categoryIcon: categoryIcon,
        model: 'Ensemble Neural Network'
    };
}

// Display prediction result
function displayPredictionResult(prediction, formData) {
    // Hide loading, show result
    document.getElementById('prediction-loading').style.display = 'none';
    document.getElementById('prediction-result').style.display = 'block';
    
    // Update result elements
    document.getElementById('confidence-score').textContent = prediction.confidence + '%';
    document.getElementById('predicted-price').textContent = prediction.price.toLocaleString();
    document.getElementById('price-category').textContent = prediction.category;
    document.getElementById('category-icon').textContent = prediction.categoryIcon;
    document.getElementById('price-min').textContent = prediction.priceRange.min.toLocaleString();
    document.getElementById('price-max').textContent = prediction.priceRange.max.toLocaleString();
    document.getElementById('price-per-carat').textContent = prediction.pricePerCarat.toLocaleString();
    
    // Show success notification
    showNotification(`Diamond price predicted: $${prediction.price.toLocaleString()}`, 'success');
}

// Save prediction to history
function saveToHistory() {
    if (predictionHistory.length > 0) {
        showNotification('Prediction saved to your history!', 'success');
    }
}

// Generate PDF report
function generateReport() {
    showNotification('PDF report generation coming soon!', 'info');
}

// Demo functionality
function showDemo() {
    showNotification('Demo video coming soon!', 'info');
}

// Notification system
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

function getNotificationIcon(type) {
    const icons = {
        'success': 'check-circle',
        'error': 'exclamation-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Loading state management
function showLoadingState(message = 'Loading...') {
    const loader = document.createElement('div');
    loader.id = 'global-loader';
    loader.className = 'global-loader';
    loader.innerHTML = `
        <div class="loader-content">
            <div class="spinner"></div>
            <p>${message}</p>
        </div>
    `;
    document.body.appendChild(loader);
}

function hideLoadingState() {
    const loader = document.getElementById('global-loader');
    if (loader) {
        loader.remove();
    }
}

// Mobile menu toggle
function toggleMobileMenu() {
    const navMenu = document.querySelector('.nav-menu');
    navMenu.classList.toggle('mobile-active');
}

// Form auto-fill for demo purposes
function fillDemoData() {
    document.getElementById('carat').value = '1.5';
    document.getElementById('cut').value = 'Ideal';
    document.getElementById('color').value = 'E';
    document.getElementById('clarity').value = 'VS1';
    document.getElementById('depth').value = '61.2';
    document.getElementById('table').value = '56.5';
    document.getElementById('x').value = '7.2';
    document.getElementById('y').value = '7.2';
    document.getElementById('z').value = '4.4';
}

// Initialize app
async function probeBackend() {
    try {
        const res = await fetch('/api/health');
        const info = await res.json();
        backendAvailable = !!info.ml_available || !!info.status;
        backendUsingMock = !!info.using_mock;
    } catch (e) {
        backendAvailable = false;
        backendUsingMock = true;
    }
}

document.addEventListener('DOMContentLoaded', async function() {
    await probeBackend();
    // Add demo data button for testing
    const demoButton = document.createElement('button');
    demoButton.textContent = 'Fill Demo Data';
    demoButton.className = 'btn btn-outline';
    demoButton.style.marginBottom = '20px';
    demoButton.onclick = fillDemoData;
    
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.parentNode.insertBefore(demoButton, predictionForm);
    }
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Escape key closes modals
        if (e.key === 'Escape') {
            document.querySelectorAll('.modal').forEach(modal => {
                modal.style.display = 'none';
            });
            document.body.style.overflow = 'auto';
        }
        
        // Ctrl/Cmd + Enter submits prediction form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const predictionForm = document.getElementById('prediction-form');
            if (predictionForm && document.getElementById('predict-section').classList.contains('active')) {
                predictionForm.dispatchEvent(new Event('submit'));
            }
        }
    });
    
    console.log('DiamondAI Frontend Initialized Successfully! 💎');
});

// Add CSS for notifications and global loader
const additionalStyles = `
<style>
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    padding: 16px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    z-index: 3000;
    min-width: 300px;
    animation: slideInRight 0.3s ease;
}

@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translateX(100%);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

.notification-success {
    border-left: 4px solid #48bb78;
}

.notification-error {
    border-left: 4px solid #f56565;
}

.notification-warning {
    border-left: 4px solid #ed8936;
}

.notification-info {
    border-left: 4px solid #4299e1;
}

.notification-content {
    display: flex;
    align-items: center;
    gap: 12px;
}

.notification-success .notification-content i {
    color: #48bb78;
}

.notification-error .notification-content i {
    color: #f56565;
}

.notification-warning .notification-content i {
    color: #ed8936;
}

.notification-info .notification-content i {
    color: #4299e1;
}

.notification-close {
    background: none;
    border: none;
    cursor: pointer;
    color: #999;
    font-size: 14px;
    padding: 4px;
}

.notification-close:hover {
    color: #333;
}

.global-loader {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(5px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 4000;
}

.loader-content {
    text-align: center;
}

.loader-content p {
    margin-top: 20px;
    font-size: 16px;
    color: #666;
}

.nav-menu.mobile-active {
    display: flex;
    flex-direction: column;
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background: white;
    border-top: 1px solid #e2e8f0;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

@media (max-width: 768px) {
    .notification {
        left: 20px;
        right: 20px;
        min-width: auto;
    }
}
</style>
`;

document.head.insertAdjacentHTML('beforeend', additionalStyles);