const playBtn = document.getElementById("playBtn");
const status = document.getElementById("status");
const channel = document.getElementById("channel");

let audio = null;

playBtn.addEventListener("click", async () => {
    status.textContent = "Generating broadcast...";
    
    const selectedChannel = channel.value;

    try {
        const response = await fetch("http://127.0.0.1:5000/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ channel: selectedChannel })
        });

        if (!response.ok) {
            status.textContent = "Backend error.";
            return;
        }

        // Receive audio as base64
        const data = await response.json();

        const audioBytes = atob(data.audio);

        const buffer = new Uint8Array(audioBytes.length);
        for (let i = 0; i < audioBytes.length; i++) {
            buffer[i] = audioBytes.charCodeAt(i);
        }

        const blob = new Blob([buffer], { type: "audio/mpeg" });
        const url = URL.createObjectURL(blob);

        // Play audio
        if (audio) {
            audio.pause();
            URL.revokeObjectURL(audio.src);
        }

        audio = new Audio(url);
        audio.play();

        status.textContent = "Now playing: " + selectedChannel.toUpperCase();

    } catch (err) {
        status.textContent = "Error connecting to server.";
        console.error(err);
    }
});
