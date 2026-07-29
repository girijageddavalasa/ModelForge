(() => {
    "use strict";
    const button = document.getElementById("predict-button");
    if (!button) return;
    button.addEventListener("click", async () => {
        const output = document.getElementById("prediction-output");
        output.classList.remove("d-none");
        try {
            const payload = JSON.parse(document.getElementById("prediction-input").value);
            const response = await fetch(button.dataset.url, {method: "POST", headers: {"Content-Type": "application/json", Accept: "application/json"}, body: JSON.stringify(payload)});
            const result = await response.json();
            output.textContent = JSON.stringify(result, null, 2);
        } catch (error) {
            output.textContent = JSON.stringify({error: error.message}, null, 2);
        }
    });
})();
