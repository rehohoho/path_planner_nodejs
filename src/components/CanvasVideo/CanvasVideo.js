import React, { Component } from 'react';
import PropTypes from 'prop-types';

import * as tf from '@tensorflow/tfjs';

class CanvasVideo extends Component {
  constructor(props) {
    //tf.enableProdMode();
    super(props);

    let value_scale = 255;
    let mean = [0.406, 0.456, 0.485];
    let std = [0.225, 0.224, 0.229];

    mean = mean.map(mean_val => mean_val * value_scale);
    std = std.map(std_val => std_val * value_scale);

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
      ]).asType('int32'),
      crop_upper_limit: 0.25,
      crop_lower_limit: 0.9,
      min_ridable_area: 0.1,
      n_slices: 20,
      preprocess_mean: mean,
      preprocess_std: std
    };
  }

  initialise_tf_constants = (width, height) => {
    
    var top_crop = Math.floor(this.state.crop_upper_limit * height);
    var bottom_crop = this.state.crop_lower_limit * height - (this.state.crop_lower_limit*height - top_crop) % this.state.n_slices;
    var crop_height = bottom_crop - top_crop;

    this.frame_constants = {
      x_offset    : tf.scalar(width/2),
      y_offset    : tf.scalar(height - 1),
      top_crop    : top_crop,
      bottom_crop : bottom_crop,
      crop_height : crop_height,
      height_idx  : tf.tidy(() => {return(
                      tf.range(top_crop, bottom_crop)
                      .reshape([this.state.n_slices, crop_height/this.state.n_slices])
                      .expandDims(2)
                    )}),
      width_idx   : tf.range(0, width),
      min_ridable_area : crop_height/this.state.n_slices * width * this.state.min_ridable_area
    }
  }

  processImage = tfimg => {
    
    tfimg = tf.tidy(() => {
      return tfimg.sub(this.state.preprocess_mean).div(this.state.preprocess_std);
    });

    return tf.transpose(tfimg, [2, 0, 1]);
  };

  get_mask = (video) => {
    // Get the image as a tensor
    const tfroadImage = tf.browser.fromPixels(video);

    const mask = tf.tidy(() => {
      const resized = tfroadImage.asType('float32')
                                // .resizeBilinear([513,513])
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

  get_waypoints_and_bestfit = (mask) => {
    
    const coords = tf.tidy(() => {
      // Sets all non-pavement to 0
      const slices = mask.mul(
                            tf.equal(mask, tf.onesLike(mask))
                          ).slice([this.frame_constants.top_crop, 0], [this.frame_constants.crop_height, this.props.options.width])
                          .reshape([this.state.n_slices, 
                            this.frame_constants.crop_height/this.state.n_slices, 
                            this.props.options.width])

      // Get waypoints
      const normalise = tf.sum(slices, [1,2]);
      const path_exist_bin_mask = tf.greater(normalise, this.frame_constants.min_ridable_area);
      
      const mid_y = tf.mul(slices, this.frame_constants.height_idx)
                      .sum([1,2])
                      .div(normalise)
                      .mul(path_exist_bin_mask);
      const mid_x = tf.mul(slices, this.frame_constants.width_idx)
                      .sum([1,2])
                      .div(normalise)
                      .mul(path_exist_bin_mask);
      
      // Get best fit through origin, OLS: xy/yy
      const trans_x = mid_x.sub( this.frame_constants.x_offset );
      const trans_y = mid_y.sub( this.frame_constants.y_offset );
      const xy      = tf.mul(trans_x, trans_y).mul(path_exist_bin_mask).sum();
      const yy      = tf.mul(trans_y, trans_y).mul(path_exist_bin_mask).sum();
      const best_fit = tf.div(xy, yy).expandDims();
      
      const coords = tf.concat([mid_x, mid_y, best_fit]);
      
      return coords;
    })

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
    canvasRef.width = this.props.options.width;
    canvasRef.height = this.props.options.height;
    this.initialise_tf_constants(canvasRef.width, canvasRef.height);
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

  draw_waypoints = (coords, context) => {
    context.strokeStyle = '#ff0000';

    context.beginPath();
    var i = 0;
    while (coords[i] <= 1) {
      i++;
    }
    context.moveTo(coords[i], coords[i+this.state.n_slices]);

    while (i < this.state.n_slices - 1) {
      if (coords[i+1] > 1) {
        context.lineTo(coords[i+1], coords[i+this.state.n_slices+1]);
      }
      i++;
    }
    context.stroke();
  }

  draw_bestfit = (grad, context) => {

    var height = this.props.options.height;
    var width = this.props.options.width;

    context.strokeStyle = '#0000ff';
    context.beginPath();
    context.moveTo(width/2, height);
    context.lineTo(grad*(-height+1) + width/2, 0)
    context.stroke();
  }

  draw = async (video, canvasRef) => {
    
    const mask = this.get_mask(video);
    const coords = await this.get_waypoints_and_bestfit(mask).data();
    // await tf.browser.toPixels(this.state.color_map.gather(mask), canvasRef)  // Uncomment to log output seg image
    
    // console.log(performance.now(), tf.memory());  // Uncomment to log tf memory usage
    
    const context = canvasRef.getContext('2d');
    const grad = coords[this.state.n_slices + this.state.n_slices];

    context.drawImage(video, 0, 0);
    if (coords.reduce((a, b) => a + b, 0) !== 1) {  // check if no waypoints are found (not enough ridable area)
      this.draw_waypoints(coords, context);
      this.draw_bestfit(grad, context);
      console.log('Suggested steer (degs to vertical): ', -Math.atan(grad) * 180 / Math.PI);
    } else {
      console.log('No waypoints detected. No steer output.')
    }
    
    // dispose tensors
    mask.dispose();

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
