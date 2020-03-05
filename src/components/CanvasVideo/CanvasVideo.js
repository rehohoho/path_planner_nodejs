import React, { Component } from 'react';
import PropTypes from 'prop-types';

import * as tf from '@tensorflow/tfjs';

class CanvasVideo extends Component {
  constructor(props) {
    //tf.enableProdMode();
    super(props);
    this.state = {
      loadingModel: true,
      color_map: tf.tensor([
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
      ]).asType('int32')
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

  predictImage = video => {
    // Get the image as a tensor
    const tfroadImage = tf.browser.fromPixels(video);
    
    const seg_map = tf.tidy(() => {
      const resized = tfroadImage.asType('float32')
                                .resizeBilinear([513, 513])
                                .reverse(-1);
      const processed = this.processImage(resized)
                            .expandDims();
      
      // Run the model on the tensor
      // No finding of main road, assumes segmentation is ok already
      const mask = this.model.predict(processed)
                               .squeeze()
                               .argMax()
      
      // Sets all non-pavement to 0
      const ridable_comparison_mask = tf.onesLike(mask)
      const ridable_area_mask = mask.clipByValue(0, 2)
                               .asType('int32')
                               .mul(
                                 tf.equal(mask, ridable_comparison_mask)
                               )

      const cropped_mask = ridable_area_mask.slice([112, 0], [400, 513])
      const slices = tf.split(cropped_mask, 5, 0)
      
      const seg_map = this.state.color_map.gather(slices[4]);

      return seg_map;

      // let x = [];
      // let y = [];

      // for (const mask_slice in slices){
      //   const normalise = tf.add(mask_slice)
      //   const mid_y = tf.add(tf.div(tf.mul(mask_slice, this.state.height_idx),normalise))
      //   const mid_x = tf.add(tf.div(tf.mul(mask_slice, this.state.width_idx),normalise))

      //   y.push(mid_y);
      //   x.push(mid_x);
      // }

      // y = tf.tensor1d(y);
      // x = tf.tensor1d(x);
    })
    
    // Tensor memory cleanup
    tfroadImage.dispose();
    
    // For testing
    // let sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
    // await sleep(10000);

    return seg_map;
  };

  componentWillMount() {
    this.virtualVideoElement = this.makeVirtualVideoElement();
  }

  componentDidMount = async () => {
    this.model = await tf.loadGraphModel(`http://127.0.0.1:81/scooter/model.json`);
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
    canvasRef.width = this.props.options.width;
    canvasRef.height = this.props.options.height;
    this.playListener = () => {
      this.draw(video, canvasRef);
    };
    video.addEventListener('play', this.playListener, false);
    if (autoplay) setTimeout(() => video.play(), 2000);
  };

  makeVirtualVideoElement = () => {
    const video = document.createElement('video');
    video.setAttribute('width', this.props.options.width);
    video.setAttribute('height', this.props.options.height);
    video.setAttribute("src", require("../../assets/whizz_video.mp4"))
    // Getting the video which has to be converted
    // navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    //   video.srcObject = stream;
    // });
    return video;
  };

  draw = async (video, canvasRef) => {
    
    if(this.props.segment) {      
      console.time('Predict');
      const seg_map = this.predictImage(video);
      tf.browser.toPixels(seg_map, canvasRef).then(() =>{
        seg_map.dispose();
      });
      console.timeEnd('Predict');
    } else {
      canvasRef.getContext('2d').drawImage(video, 0, 0);
    }
    
    if (!video.paused && !video.ended) {
      setTimeout(this.draw, 1000 / 24, video, canvasRef);
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
