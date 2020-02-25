import React, { Component } from 'react';
import PropTypes from 'prop-types';

import * as tf from '@tensorflow/tfjs';

class CanvasVideo extends Component {
  constructor(props) {
    //tf.enableProdMode();
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

  processImage = tfimg => {
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

  predictImage = async video => {
    // Get the image as a tensor
    let tfroadImage = tf.browser.fromPixels(video);
    tfroadImage = tf.image.resizeBilinear(tfroadImage, [513, 513]);
    tfroadImage = tf.reverse(tfroadImage, -1);
    tfroadImage = this.processImage(tfroadImage);
    tfroadImage.array().then(data => {});
    const resized = tf.cast(tfroadImage, 'float32');
    const roadPixels = tf.tensor4d(Array.from(await resized.data()), [1,3,513,513]);

    // Run the model on the tensor
    let predictions = tf
      .tensor1d(await this.model.predict(roadPixels).data())
      .reshape([19, 513, 513])
      .argMax();

    roadPixels.dispose();

    predictions.array().then(data => {});

    // Convert tensor to array and assign color to each pixel
    const segMap = Array.from(await predictions.data());
    const segMapColor = segMap.map(seg => this.state.colors[seg]);

    predictions.dispose()

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

    return imageData;
  };

  componentWillMount() {
    this.virtualVideoElement = this.makeVirtualVideoElement();
  }

  componentDidMount = async () => {
    this.model = await tf.loadGraphModel(`http://127.0.0.1:81/model.json`);
    this.setState({ loadingModel: false });

    this.startPlayingInCanvas(this.virtualVideoElement, this.canvasRef, {
      autoplay: this.props.options
        ? this.props.options.autoplay || false
        : false
    });
  };

  componentWillUnmount() {
    this.virtualVideoElement.removeEventListener(
      'play',
      this.playListener,
      false
    );
    this.virtualVideoElement.remove();
    delete this.virtualVideoElement;
  }

  startPlayingInCanvas = (video, canvasRef, { ratio, autoplay }) => {
    const context = canvasRef.getContext('2d');
    canvasRef.width = this.props.options.width;
    canvasRef.height = this.props.options.height;
    this.playListener = () => {
      this.draw(video, context, canvasRef.width, canvasRef.height);
    };
    video.addEventListener('play', this.playListener, false);
    if (autoplay) setTimeout(() => video.play(), 2000);
  };

  makeVirtualVideoElement = () => {
    const video = document.createElement('video');
    video.setAttribute('width', this.props.options.width);
    video.setAttribute('height', this.props.options.height);
    // video.setAttribute("src", require("../../assets/video.mp4"))
    // Getting the video which has to be converted
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
      video.srcObject = stream;
    });
    return video;
  };

  draw = async (video, context) => {
    
    if(this.props.segment) {
      console.time('predict');
      const processed = await this.predictImage(video);
      console.timeEnd('predict');
      context.putImageData(processed, 0, 0);
    } else {
      context.drawImage(video, 0, 0)
    }
    
    if (!video.paused && !video.ended) {
      setTimeout(this.draw, 1000 / 24, video, context);
    }
  };

  onPlayPauseHandler = e => {
    this.virtualVideoElement.paused
      ? this.virtualVideoElement.play()
      : this.virtualVideoElement.pause();
  };

  render() {
    return (
      <div>
        <canvas
          ref={canvasRef => (this.canvasRef = canvasRef)}
          onClick={this.onPlayPauseHandler}
        ></canvas>
      </div>
    );
  }
}

CanvasVideo.propTypes = {
  options: PropTypes.shape({
    autoplay: PropTypes.bool,
    width: PropTypes.number,
    height: PropTypes.number
  })
};

export default CanvasVideo;
