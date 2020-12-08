let handpose;
let video;
let predictions = [];

let model;
let targetLabel;
let state = 'collection';
// let state = 'prediction';

let socket;

let nnResults;
let loopBroken = false;

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.size(width, height);

  socket = io.connect();

  handpose = ml5.handpose(video, modelReady);
  // This sets up an event that fills the global variable "predictions"
  // with an array every time new hand poses are detected
  handpose.on('predict', results => {
    predictions = results;
  });

  let options = {
    inputs: 63,
    outputs: ['label'],
    task: 'classification',
    debug: 'false',
  };

  model = ml5.neuralNetwork(options);
  autoStartPredict();

  // Hide the video element, and just show the canvas
  video.hide();

  createButton('Load Model').mousePressed(onLoadModelClick);
  createButton('Start Prediction').mousePressed(onPredictClick);
}

function autoStartPredict() {
  if (state == 'prediction') {
    onLoadModelClick();
    onPredictClick();
  }
}

function dataLoaded() {
  console.log(model.data);
}

function modelReady() {
  console.log('Model ready!');
}

function draw() {
  image(video, 0, 0, width, height);

  // We can call both functions to draw all keypoints
  drawKeypoints();
  restartPredictions();
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  for (let i = 0; i < predictions.length; i += 1) {
    const prediction = predictions[i];

    for (let j = 0; j < prediction.landmarks.length; j += 1) {
      const keypoint = prediction.landmarks[j];
      fill(0, 255, 0);
      noStroke();
      ellipse(keypoint[0], keypoint[1], 10, 10);
    }
  }
}

function mousePressed() {
  if (predictions[0] != undefined) {
    let inputs = predictions[0].landmarks.flat();
    console.log(predictions[0].landmarks.flat());
    if (state == 'collection') {
      let target = {
        label: targetLabel,
      };
      if (targetLabel != undefined) {
        model.addData(inputs, target);
        console.log(`Data recorded for label ${targetLabel}`);
      } else {
        console.log('Target label not set.');
      }
      text(targetLabel, mouseX, mouseY);
    } else if (state == 'prediction') {
      model.classify(inputs, gotResults);
    }
  }
}

function gotResults(error, results) {
  if (error) {
    console.error(error);
    return;
  }
  console.log(`${results[0].label}: ${results[0].confidence}`); // print label & confidence
  nnResults = results;
  classify();
}

function keyPressed() {
  if (key == 't') {
    console.log('starting training');
    state = 'training';
    model.normalizeData();
    let options = {
      epochs: 50,
    };
    model.train(options, whileTraining, finishedTraining);
  } else if (key == 's') {
    model.saveData();
  } else if (key == 'm') {
    model.save();
  } else {
    targetLabel = key.toUpperCase();
  }
}

function whileTraining(epoch, loss) {
  console.log(epoch, loss);
}
function finishedTraining() {
  console.log('finished training');
}

function classify() {
  if (predictions[0] != undefined) {
    let inputs = predictions[0].landmarks.flat();
    model.classify(inputs, gotResults);
  } else {
    loopBroken = true;
  }
}

function onPredictClick() {
  state = 'prediction';
}
function onLoadModelClick() {
  const modelInfo = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin',
  };
  model.load(modelInfo, () => console.log('Model Loaded.'));
}

function restartPredictions() {
  if (loopBroken) {
    loopBroken = false;
    classify();
  }
}

const sendToServer = () => {
  socket.emit('handpose', nnResults);
};
