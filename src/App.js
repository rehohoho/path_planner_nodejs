import React from 'react';
import './App.css';
import img from './assets/car.jpg';
import img2 from './assets/road.jpg';

import CanvasVideo from './components/CanvasVideo/CanvasVideo';

import * as tf from '@tensorflow/tfjs';

class App extends React.Component {
  constructor(props) {
    tf.enableProdMode();
    super(props);
    this.state = {
      loadingModel: true,
      colors: [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]
      ]
    };
  }

  //CHANGED
  image_processing = tfimg => {
    let value_scale = 255;
    let mean = [0.406, 0.456, 0.485];
    let std = [0.225, 0.224, 0.229];

    mean = mean.map(mean_val => mean_val * value_scale);
    std = std.map(std_val => std_val * value_scale);

    tfimg = tf.tidy(() => {
      return tfimg.sub(mean).div(std);
    });

    return tf.transpose(tfimg, [2, 0, 1]);
  };

  async componentDidMount() {
    // Get the model
    this.model = await tf.loadGraphModel(`http://127.0.0.1:81/model.json`);
    this.setState({ loadingModel: false });
  }

  predictImage = async name => {
    // Get the image as a tensor
    let tfroadImage = tf.browser.fromPixels(document.getElementById(name));

    //CHANGED
    tfroadImage = tf.image.resizeBilinear(tfroadImage, [513, 513]);

    //CHANGED
    tfroadImage = tf.reverse(tfroadImage, -1);
    tfroadImage = this.image_processing(tfroadImage);
    tfroadImage.array().then(data => {});
    const resized = tf.cast(tfroadImage, 'float32');
    const roadPixels = tf.tensor4d(Array.from(await resized.data()), [
      1,
      3,
      513,
      513
    ]);

    // Run the model on the tensor
    let predictions = tf
      .tensor1d(await this.model.predict(roadPixels).data())
      .reshape([19, 513, 513])
      .argMax();

    // Dispose of tensor
    roadPixels.dispose();

    predictions.array().then(data => {});

    // Convert tensor to array and assign color to each pixel
    const segMap = Array.from(await predictions.data());
    const segMapColor = segMap.map(seg => this.state.colors[seg]);

    // Dispose of tensor
    predictions.dispose();

    // Convert array to data for image
    let data = [];
    for (var i = 0; i < segMapColor.length; i++) {
      data.push(segMapColor[i][0]); // red
      data.push(segMapColor[i][1]); // green
      data.push(segMapColor[i][2]); // blue
      data.push(255); // alpha
    }

    // Create ImageData
    let imageData = new ImageData(513, 513);
    imageData.data.set(data);

    // Show ImageData on canvas
    document
      .getElementById('result')
      .getContext('2d')
      .putImageData(imageData, 0, 0);
  };

  render() {
    return (
      <div>
        {/* <img src={img} id="car" width="513" height="513" alt="car" />
        <img src={img2} id="road" width="513" height="513" alt="road" />
        <canvas id="result" width="513" height="513"></canvas>
        {this.state.loadingModel ? (
          <h1>Model is loading</h1>
        ) : (
          <button
            onClick={async () => {
              console.time('Inference 1');
              await this.predictImage('car');
              console.timeEnd('Inference 1');
            }}
          >
            Segment Car Image
          </button>
        )}
        {this.state.loadingModel ? (
          <h1>Model is loading</h1>
        ) : (
          <button
            onClick={async () => {
              console.time('Inference 1');
              await this.predictImage('road');
              console.timeEnd('Inference 1');
            }}
          >
            Segment Road Image
          </button>
        )} */}
        {/* <WebcamStream></WebcamStream> */}
        <CanvasVideo options={{ autoplay: true, width: 513, height: 513 }} segment={true}/>
        <CanvasVideo options={{ autoplay: true, width: 513, height: 513 }} segment={false}/>
      </div>
    );
  }
}

export default App;
