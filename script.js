const canvas = document.getElementById("ar-scene");
const ctx = canvas.getContext("2d");

let width = 0;
let height = 0;
let tick = 0;

function resize() {
  const ratio = window.devicePixelRatio || 1;
  width = canvas.clientWidth;
  height = canvas.clientHeight;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function line(x1, y1, x2, y2, color, alpha = 1) {
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawShip(baseY) {
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#d9952f";
  ctx.fillStyle = "rgba(217, 149, 47, 0.08)";
  ctx.beginPath();
  ctx.moveTo(width * 0.14, baseY);
  ctx.lineTo(width * 0.76, baseY);
  ctx.lineTo(width * 0.68, baseY + 76);
  ctx.lineTo(width * 0.22, baseY + 76);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  for (let i = 0; i < 4; i += 1) {
    const x = width * 0.28 + i * width * 0.095;
    ctx.strokeRect(x, baseY - 76, width * 0.045, 76);
    line(x + width * 0.022, baseY - 96, x + width * 0.022, baseY - 76, "#d9952f", 0.9);
  }
}

function draw() {
  tick += 0.012;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#071017";
  ctx.fillRect(0, 0, width, height);

  ctx.lineWidth = 1;
  for (let y = 80; y < height; y += 42) {
    const offset = Math.sin(tick + y * 0.01) * 16;
    line(0, y + offset, width, y - offset, "#008b8b", 0.12);
  }

  for (let x = -80; x < width + 80; x += 86) {
    line(x, 0, x + Math.sin(tick + x) * 60, height, "#204d7a", 0.16);
  }

  const baseY = height * 0.62 + Math.sin(tick * 1.4) * 8;
  drawShip(baseY);

  ctx.fillStyle = "rgba(0, 139, 139, 0.78)";
  for (let i = 0; i < 32; i += 1) {
    const x = (i * 97 + tick * 140) % (width + 120) - 60;
    const y = 120 + ((i * 53) % Math.max(160, height - 220));
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.font = "700 14px Arial";
  ctx.fillStyle = "rgba(248, 251, 252, 0.72)";
  ctx.fillText("P(Survived)", width * 0.68, height * 0.28);
  ctx.fillText("Sex | Age | Pclass | Fare", width * 0.58, height * 0.34);
  ctx.fillText("CRISP-ML", width * 0.73, height * 0.72);

  requestAnimationFrame(draw);
}

window.addEventListener("resize", resize);
resize();
draw();
