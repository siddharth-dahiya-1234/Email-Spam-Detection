const form = document.getElementById("prediction-form");

const resultContainer = document.getElementById("result");

const emailContent = document.getElementById("email-input");

const apiUrl = "http://localhost:8000/predict"

async function predict(emailContext) {
    const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: emailContext })
    })

    if (!response.ok) {
        throw new Error("Network response was not ok");
    }
    const data = await response.json();
    return data;
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const emailContext = emailContent.value;
    try {
        const data = await predict(emailContext);
        const prediction = data.predictions[0];

        resultContainer.style.display = "block";
        resultContainer.innerHTML = `
            <h2>Prediction Result</h2>
            <p><strong>Probability:</strong> ${(prediction.probability * 100).toFixed(2)}%</p>
            <p><strong>Predicted Label:</strong> ${prediction.label}</p>
        `;
    } catch (error) {
        console.error("Error:", error);
    }
})