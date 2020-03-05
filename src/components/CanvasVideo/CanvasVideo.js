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
      // slice_height_range: tf.expandDims(tf.range(0, 80), 1), // 80 = slice height
      // slice_width_range: tf.range(0, 513)                     // 513 = slice width
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

  get_mask = (video) => {
    // Get the image as a tensor
    const tfroadImage = tf.browser.fromPixels(video);

    const mask = tf.tidy(() => {
      const resized = tfroadImage.asType('float32')
                                .reverse(-1);
      const processed = this.processImage(resized)
                            .expandDims();
      // Run the model on the tensor
      // No finding of main road, assumes segmentation is ok already
      const mask = this.model.predict(processed)
                               .squeeze()
                               .argMax()
      return mask;
    })

    tfroadImage.dispose();
    return mask;
  }

  get_waypoints = (mask, n_slices) => {
    
    const height = this.props.options.height;
    const width = this.props.options.width;

    const coords = tf.tidy(() => {
      // Sets all non-pavement to 0
      const ridable_area_mask = mask.mul(
                                  tf.equal(mask, tf.onesLike(mask))
                                ).asType("float32");
      const slices = tf.stack(
                        tf.split(ridable_area_mask, n_slices, 0)
                     );
      
      // Get waypoints
      const normalise = tf.sum(slices, [1,2]);
      const height_idx = tf.range(0, height)
                        .reshape([n_slices, height/n_slices])
                        .expandDims(2);
      const width_idx = tf.range(0, width);
      const mid_y = tf.div(tf.sum(tf.mul(slices, height_idx), [1,2]), normalise);
      const mid_x = tf.div(tf.sum(tf.mul(slices, width_idx), [1,2]), normalise);
      const coords = tf.stack([mid_x, mid_y])
                     .asType("int32");

      return(coords)
    })

    // let x = [];
    // let y = [];
    // let rolling_height_idx = 0;

    // for (const i in slices){
    //   const mask_slice = slices[i];
    //   const normalise = tf.sum(mask_slice);
    //   if (normalise.notEqual(0).dataSync()[0]) {
    //     const height_idx = this.state.slice_height_range.add(rolling_height_idx)
    //     const mid_y = tf.sum(tf.div(tf.mul(mask_slice, height_idx), normalise))
    //     const mid_x = tf.sum(tf.div(tf.mul(mask_slice, this.state.slice_width_range), normalise))
    //     // y = np.sum( a* np.expand_dims(np.arange(0,len(a)), 1)) / np.sum(a)
    //     // x = np.sum( a* np.arange(0,len(a))) / np.sum(a)
    //     // y = tf.reduce_sum(tf.divide(tf.multiply(a, tf.expand_dims(tf.range(0, a.shape[0]), 1)), tf.reduce_sum(a)))
    //     // x = tf.reduce_sum(tf.divide(tf.multiply(a, tf.range(0, a.shape[0])), tf.reduce_sum(a)))
    //     y.push(mid_y.dataSync());
    //     x.push(mid_x.dataSync());
    //   }
    //   rolling_height_idx += 80;
    // }
    
    // For testing
    // let sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
    // await sleep(10000);

    return coords;
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
    // const context = canvasRef.getContext('2d');
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
    
    // if(this.props.segment) {      
    //   console.time('Predict');
    //   const seg_map = this.get_mask(video);
    //   seg_map.dispose();
    //   // tf.browser.toPixels(seg_map, canvasRef).then(() =>{
    //   //   seg_map.dispose();
    //   // });
    //   console.timeEnd('Predict');
    // } else {
    //   canvasRef.getContext('2d').drawImage(video, 0, 0);
    // }
    
    const n_slices = 20;
    const mask = this.get_mask(video);
    const coords = this.get_waypoints(mask, n_slices);
    await tf.browser.toPixels(this.state.color_map.gather(mask), canvasRef)
    // context.drawImage(video, 0, 0);

    const context = canvasRef.getContext('2d');
    context.beginPath();
    context.strokeStyle = '#ff0000';
    context.lineWidth = 5;
    
    console.log(coords);

    var i = 0;
    while (coords[i] == 1) {
      i++;
    }
    context.moveTo(coords[i], coords[i+n_slices]);

    while (i < n_slices - 1) {
      if (coords[i+1] != 1) {
        context.lineTo(coords[i+1], coords[i+n_slices+1]);
      }
      i++;
    }
    context.lineTo(this.props.options.width/2, this.props.options.height);
    context.stroke();
    
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
