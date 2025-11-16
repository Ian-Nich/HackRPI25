const playBtn = document.getElementById("playBtn");
const status = document.getElementById("status");
const channel = document.getElementById("channel");
const cassetteContainer = document.getElementById("cassetteContainer");
const waveContainer = document.getElementById("waveContainer");
const transcriptText = document.getElementById("transcriptText");

let audio = null;
let transcriptTimer = null;
const TRANSCRIPT_CHAR_INTERVAL_MS = 55;

function startAnimations() {
    cassetteContainer.classList.add("playing");
    waveContainer.classList.add("playing");
}

function stopAnimations() {
    cassetteContainer.classList.remove("playing");
    waveContainer.classList.remove("playing");
}

function setTranscriptInstant(text) {
    if (!transcriptText) return;
    if (transcriptTimer) {
        clearInterval(transcriptTimer);
        transcriptTimer = null;
    }
    transcriptText.textContent = text;
}

function typeTranscript(text) {
    if (!transcriptText) return;
    if (transcriptTimer) {
        clearInterval(transcriptTimer);
        transcriptTimer = null;
    }
    transcriptText.textContent = "";
    const chars = text.split("");
    let idx = 0;
    transcriptTimer = setInterval(() => {
        transcriptText.textContent += chars[idx];
        idx += 1;
        if (idx >= chars.length) {
            clearInterval(transcriptTimer);
            transcriptTimer = null;
        }
    }, TRANSCRIPT_CHAR_INTERVAL_MS);
}

playBtn.addEventListener("click", async () => {
    status.textContent = "Generating broadcast...";
    setTranscriptInstant("Tuning into " + channel.options[channel.selectedIndex].text + "...");

    const selectedChannel = channel.value;

    try {
        const response = await fetch("http://127.0.0.1:5000/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ station: selectedChannel })
        });

        const data = await response.json();
        console.log("Backend response:", data);

        if (!response.ok) {
            const errorMsg = data.error || "Backend error.";
            status.textContent = `Error: ${errorMsg}`;
            console.error("Backend error:", data);
            setTranscriptInstant(`Transmission error: ${errorMsg}`);
            return;
        }

        if (!data.audio) {
            status.textContent = "No audio returned.";
            console.error(data);
            setTranscriptInstant("No audio returned from the DJ console.");
            return;
        }

        if (data.script) {
            typeTranscript(data.script);
        } else {
            setTranscriptInstant("Transcript unavailable for this broadcast.");
        }

        // Stop previous audio and animations
        if (audio) {
            audio.pause();
            stopAnimations();
        }

        // Create and play new audio
        audio = new Audio("data:audio/mp3;base64," + data.audio);
        
        // Start animations when audio starts playing
        audio.addEventListener("play", () => {
            startAnimations();
            status.textContent = "Now playing: " + selectedChannel.toUpperCase();
        });
        
        // Stop animations when audio ends or is paused
        audio.addEventListener("ended", () => {
            stopAnimations();
            status.textContent = "Broadcast ended. Select a channel...";
            setTranscriptInstant(transcriptText.textContent + "\n\n// Transmission complete //");
        });
        
        audio.addEventListener("pause", () => {
            stopAnimations();
        });
        
        audio.play();

    } catch (err) {
        status.textContent = "Error connecting to server.";
        console.error(err);
        setTranscriptInstant("Signal lost. Please try again.");
    }
});
