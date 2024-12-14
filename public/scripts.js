var operation = "sum";
var operation_sign;
var canvas1;
var canvas2;
var calculate;
var result;

var drawing = false;
var canvas1InitialXY = { x: 0, y: 0 };
var canvas2InitialXY = { x: 0, y: 0 };

document.addEventListener("DOMContentLoaded", () => {
  getElements();
  assignEvents();
  const context1 = canvas1.getContext("2d");
  const context2 = canvas2.getContext("2d");

  canvas1.width = canvas2.width = 149;
  canvas1.height = canvas2.height = 149;
});

function getElements() {
  canvas1 = document.getElementById("canvasRef1");
  canvas2 = document.getElementById("canvasRef2");
  operation_sign = document.getElementById("operation_sign");
  calculate = document.getElementById("calculate");
  result = document.getElementById("result");
}

function assignEvents() {
  document
    .getElementById("operation_add")
    .addEventListener("click", () => setOperation("sum"));

  document
    .getElementById("operation_subtract")
    .addEventListener("click", () => setOperation("subtract"));

  document
    .getElementById("operation_multiply")
    .addEventListener("click", () => setOperation("multiply"));

  document
    .getElementById("operation_divide")
    .addEventListener("click", () => setOperation("divide"));

  calculate.addEventListener("click", onCalculate);

  setupCanvas();
}

function setupCanvas() {
  canvas1.addEventListener("mousedown", (e) => {
    drawing = true;
    canvas1InitialXY = { x: e.offsetX, y: e.offsetY };
  });

  canvas1.addEventListener("mouseup", (e) => {
    drawing = false;
  });

  canvas1.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    canvas1InitialXY = { x: e.offsetX, y: e.offsetY };
    drawCanvas(
      canvas1.getContext("2d"),
      canvas1InitialXY,
      e.offsetX,
      e.offsetY
    );
  });

  document.getElementById("clear1").addEventListener("click", () => {
    clearCanvas(canvas1.getContext("2d"), canvas1);
  });

  canvas2.addEventListener("mousedown", (e) => {
    drawing = true;
    canvas2InitialXY = { x: e.offsetX, y: e.offsetY };
  });

  canvas2.addEventListener("mouseup", (e) => {
    drawing = false;
  });

  canvas2.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    canvas2InitialXY = { x: e.offsetX, y: e.offsetY };
    drawCanvas(
      canvas2.getContext("2d"),
      canvas2InitialXY,
      e.offsetX,
      e.offsetY
    );
  });

  document.getElementById("clear1").addEventListener("click", () => {
    clearCanvas(canvas1.getContext("2d"));
  });

  document.getElementById("clear2").addEventListener("click", () => {
    clearCanvas(canvas2.getContext("2d"));
  });
}

function setOperation(_operation) {
  operation = _operation;
  switch (_operation) {
    case "sum":
      operation_sign.innerHTML = "+";
      break;
    case "subtract":
      operation_sign.innerHTML = "-";
      break;
    case "multiply":
      operation_sign.innerHTML = "*";
      break;
    case "divide":
      operation_sign.innerHTML = "/";
      break;
    default:
      operation_sign.innerHTML = "+";
      break;
  }
}

async function onCalculate() {
  // Obtener los blobs de las imágenes desde los canvas
  const imagen1Blob = await new Promise((resolve) =>
    canvas1.toBlob(resolve, "image/png")
  );
  const imagen2Blob = await new Promise((resolve) =>
    canvas2.toBlob(resolve, "image/png")
  );

  // Crear un objeto FormData para enviar las imágenes y la operación
  const formData = new FormData();
  formData.append("images", imagen1Blob, "imagen1.png");
  formData.append("images", imagen2Blob, "imagen2.png");
  formData.append("operation", operation);

  // Realizar la solicitud POST al backend
  try {
    const response = await fetch("/api/v1/calculate", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log(data);
    result.innerHTML = data.result;
  } catch (error) {
    console.error("Error:", error);
  }
}

function drawCanvas(context, canvasInitial = { x: 0, y: 0 }, x = 0, y = 0) {
  context.beginPath();
  context.lineWidth = 5;
  context.strokeStyle = "#000000";
  context.lineCap = "round";
  context.lineJoin = "round";
  context.moveTo(canvasInitial.x, canvasInitial.y);
  context.lineTo(x, y);
  context.stroke();
  canvasInitial.x = x;
  canvasInitial.y = y;
}

function clearCanvas(context) {
  context.clearRect(0, 0, 149, 149);
}
