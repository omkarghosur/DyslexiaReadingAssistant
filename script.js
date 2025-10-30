const backend = "http://127.0.0.1:8000";

document.getElementById("detectBtn").onclick = async () => {
  const res = await fetch(`${backend}/detect`);
  const data = await res.json();
  document.getElementById("objectName").innerText = `Detected Object: ${data.object}`;
};

document.getElementById("speakBtn").onclick = async () => {
  const object = document.getElementById("objectName").innerText.split(": ")[1];
  if (object && object !== "—") {
    await fetch(`${backend}/speak?word=${object}`);
  }
};

document.getElementById("checkBtn").onclick = async () => {
  const object = document.getElementById("objectName").innerText.split(": ")[1];
  if (object && object !== "—") {
    const res = await fetch(`${backend}/check_pronunciation?word=${object}`);
    const data = await res.json();

    if (data.error) {
      document.getElementById("feedback").innerText = "⚠️ " + data.error;
    } else {
      document.getElementById("feedback").innerText =
        "🗣 You said: " + data.spoken_text + "\nFeedback: " + JSON.stringify(data.feedback);
    }
  }
};
