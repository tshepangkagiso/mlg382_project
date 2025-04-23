document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultPlaceholder = document.getElementById('resultPlaceholder');
    const predictionResult = document.getElementById('predictionResult');
    const predictionText = document.getElementById('predictionText');
    const probabilityText = document.getElementById('probabilityText');
    const riskBar = document.getElementById('riskBar');
    const resultIcon = document.getElementById('resultIcon');
    const recommendationsList = document.getElementById('recommendations');

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading state
        resultPlaceholder.innerHTML = `
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3">Analyzing customer data...</p>
        `;
        
        // Collect form data
        const formData = {
            gender: document.getElementById('gender').value,
            seniorCitizen: document.getElementById('seniorCitizen').value,
            multipleLines: document.getElementById('multipleLines').value,
            internetService: document.getElementById('internetService').value,
            contract: document.getElementById('contract').value,
            paperlessBilling: document.getElementById('paperlessBilling').value,
            paymentMethod: document.getElementById('paymentMethod').value,
            hasFamily: document.getElementById('hasFamily').value,
            monthlyCost: document.getElementById('monthlyCost').value
        };
        
        // Send to backend
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                displayResults(data);
            } else {
                showError(data.message);
            }
        })
        .catch(error => {
            showError('An error occurred while processing your request.');
            console.error('Error:', error);
        });
    });
    
    function displayResults(data) {
        // Hide placeholder, show results
        resultPlaceholder.classList.add('d-none');
        predictionResult.classList.remove('d-none');
        predictionResult.classList.add('fade-in');
        
        // Set prediction text
        predictionText.textContent = data.prediction === 'Yes' ? 'High Churn Risk' : 'Low Churn Risk';
        
        // Calculate probability percentage
        const probability = Math.round(data.probability * 100);
        probabilityText.textContent = probability;
        
        // Set visual elements based on prediction
        if (data.prediction === 'Yes') {
            resultIcon.className = 'bi bi-exclamation-triangle-fill text-danger';
            predictionText.className = 'mt-3 text-danger';
            riskBar.className = 'progress-bar bg-danger';
            riskBar.style.width = `${probability}%`;
            riskBar.setAttribute('aria-valuenow', probability);
        } else {
            resultIcon.className = 'bi bi-check-circle-fill text-success';
            predictionText.className = 'mt-3 text-success';
            riskBar.className = 'progress-bar bg-success';
            riskBar.style.width = `${probability}%`;
            riskBar.setAttribute('aria-valuenow', probability);
        }
        
        // Generate recommendations
        generateRecommendations(data);
    }
    
    function generateRecommendations(data) {
        recommendationsList.innerHTML = '';
        
        if (data.prediction === 'Yes') {
            // High churn risk recommendations
            recommendationsList.innerHTML = `
                <li>Offer loyalty discount or promotional rate</li>
                <li>Suggest contract renewal with incentives</li>
                <li>Propose upgraded service package</li>
                <li>Assign dedicated account manager</li>
                <li>Conduct satisfaction survey</li>
            `;
        } else {
            // Low churn risk recommendations
            recommendationsList.innerHTML = `
                <li>Maintain regular communication</li>
                <li>Offer value-added services</li>
                <li>Provide occasional service check-ins</li>
                <li>Consider referral program incentives</li>
            `;
        }
    }
    
    function showError(message) {
        resultPlaceholder.innerHTML = `
            <i class="bi bi-x-circle-fill text-danger"></i>
            <p class="mt-3">${message}</p>
            <button class="btn btn-sm btn-outline-primary mt-2" onclick="window.location.reload()">Try Again</button>
        `;
    }
});