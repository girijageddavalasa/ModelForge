(() => {
    "use strict";
    const container = document.getElementById("training-job");
    if (!container || !["queued", "running"].includes(container.dataset.currentStatus)) return;
    const statusElement = document.getElementById("job-status");
    const progressBar = document.getElementById("progress-bar");
    const progressLabel = document.getElementById("progress-label");
    const poll = async () => {
        try {
            const response = await fetch(container.dataset.statusUrl, {headers: {Accept: "application/json"}});
            if (!response.ok) throw new Error("Status request failed");
            const job = await response.json();
            statusElement.textContent = job.status;
            progressBar.style.width = `${job.progress}%`;
            progressLabel.textContent = `${job.progress}%`;
            if (["completed", "failed", "cancelled"].includes(job.status)) {
                window.location.reload();
                return;
            }
        } catch (error) {
            console.error("Unable to refresh training progress", error);
        }
        window.setTimeout(poll, 1000);
    };
    window.setTimeout(poll, 500);
})();
