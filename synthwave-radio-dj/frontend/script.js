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
            body: JSON.stringify({ station: selectedChannel })
        });

        if (!response.ok) {
            status.textContent = "Backend error.";
            return;
        }

const data = await response.json();
console.log("Backend response:", data);

if (!data.audio) {
    status.textContent = "No audio returned.";
    console.error(data);
    return;
}

if (audio) {
    audio.pause();
}

audio = new Audio("data:audio/mp3;base64," + data.audio);
audio.play();

status.textContent = "Now playing: " + selectedChannel.toUpperCase();

    } catch (err) {
        status.textContent = "Error connecting to server.";
        console.error(err);
    }
});
