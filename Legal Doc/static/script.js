document.addEventListener("DOMContentLoaded", function () {
    fetchLanguages();
});

async function fetchLanguages() {
    let dropdown = document.getElementById("languageSelect");
    let languages = {
        "en": "English", "hi": "Hindi", "fr": "French", "es": "Spanish",
        "ta": "Tamil", "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati",
        "te": "Telugu", "ur": "Urdu", "pa": "Punjabi"
    };

    Object.keys(languages).forEach(code => {
        let option = document.createElement("option");
        option.value = code;
        option.textContent = languages[code];
        dropdown.appendChild(option);
    });
}

document.getElementById("uploadForm").addEventListener("submit", function (event) {
    event.preventDefault();
    let fileInput = document.getElementById("fileInput");
    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    document.getElementById("summary").textContent = "Analyzing document...";
    document.getElementById("classification").textContent = "";
    document.getElementById("riskScore").textContent = "";

    fetch("/analyze", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("summary").textContent = data.summary;
        document.getElementById("classification").textContent = data.classification;
        document.getElementById("riskScore").textContent = data.risk_score;
        document.getElementById("results").style.display = "block";
    })
    .catch(error => console.error("Error:", error));
});

function translateText() {
    let textToTranslate = document.getElementById("summary").textContent + " " + document.getElementById("classification").textContent;
    let selectedLanguage = document.getElementById("languageSelect").value;

    fetch("/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textToTranslate, language: selectedLanguage })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("translatedText").textContent = data.translated_text;
    })
    .catch(error => console.error("Translation Error:", error));
}

function askChatbot() {
    let question = document.getElementById("questionInput").value.trim();
    let documentText = document.getElementById("summary").textContent;

    if (question === "") {
        document.getElementById("chatbotResponse").textContent = "Please enter a question.";
        return;
    }

    fetch("/chatbot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question, text: documentText })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("chatbotResponse").textContent = data.answer;
    })
    .catch(error => console.error("Chatbot Error:", error));
}

function downloadReport() {
    let summary = document.getElementById("summary").textContent;
    let classification = document.getElementById("classification").textContent;
    let riskScore = document.getElementById("riskScore").textContent;

    fetch("/download_report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
            summary: summary, 
            classification: classification, 
            risk_score: riskScore 
        })
    })
    .then(response => response.blob())
    .then(blob => {
        let url = window.URL.createObjectURL(blob);
        let a = document.createElement("a");
        a.href = url;
        a.download = "Legal_Report.pdf";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    })
    .catch(error => console.error("Download Error:", error));
}
