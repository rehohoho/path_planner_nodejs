import React, { Component } from 'react';
import PropTypes from 'prop-types';

import * as tf from '@tensorflow/tfjs';

class pid {
  // pid class with reference from python simple_pid library

  kp = 1.0;
  ki = 0.0;
  kd = 0.0;
  setpoint = 0.0;
  upper_limit = null;
  lower_limit = null;

  constructor(kp, ki, kd, setpoint, upper_limit, lower_limit) {
    this.kp = kp;
    this.ki = ki;
    this.kd = kd;
    this.setpoint = setpoint;
    this.upper_limit = upper_limit;
    this.lower_limit = lower_limit;
    
    this._reset();

    this.auto_mode = true;
  }

  _clamp = (value) => {
    if (this.upper_limit != null && value > this.upper_limit) {
      return this.upper_limit;
    }
    if (this.lower_limit != null && value < this.lower_limit) {
      return this.lower_limit;
    }
    return value;
  }

  _reset = () => {
    this.p = 0.0;
    this.i = 0.0;
    this.d = 0.0;

    this.last_time = Date.now() / 1000;
    this.last_output = null;
    this.last_feedback = null;
  }

  update = (feedback) => {
    
    if (!this.auto_mode){
      if (this.last_output == null){
        console.log("PID has not done its first iteration. Toggle auto_mode before first update.");
      }
      return this.last_output;
    }

    let now = Date.now() / 1000 // time in seconds
    
    let dt = 1e16; // obtain change in time
    if (this.last_time != null) {
      dt = now - this.last_time;
    }

    let d_feedback = feedback; // obtain change in feedback
    if (this.last_feedback != null) {
      d_feedback = feedback - this.last_feedback;
    }

    let error = this.setpoint - feedback; // compute pid terms
    this.p = this.kp * error;
    this.i += this.ki * error*dt;
    this.i = this._clamp(this.i); // avoid integral windup
    this.d = -this.kd * d_feedback / dt;
    
    let output = this.p + this.i + this.d;
    output = this._clamp(output)
    // console.log(this.p, this.i, this.d);

    this.last_feedback = feedback; // record states for next update
    this.last_time = now;
    this.last_output = output;

    return output;
  }

  toggle_auto_mode = () => {
    if (!this.auto_mode) { // from manual to auto requires reset of parameters
      this._reset();
      if (this.last_output != null) {
        this.i = this.last_output;
      } else {
        this.i = 0.0;
      }
      this.i = this._clamp(this.i);
    }

    this.auto_mode = !this.auto_mode;
  }
}

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
      preprocess_std: std,
      model_height: 513,
      model_width: 513,
    };

    this.controller_state = {
      pers_poly: [7.938e-05, -0.002267, 0.5619, -0.02184],
      steer_max: 880,
      steer_mid: 445,
      steer_min: 10,
      step_per_degree: 10,
      pid: new pid(0.85, 0.0, 0.4, 0.0, 44.0, -44.0)
    };
  }

  initialise_tf_constants = (width, height) => {
    
    var top_crop = Math.floor(this.state.crop_upper_limit * height);
    var bottom_crop = this.state.crop_lower_limit * height - (this.state.crop_lower_limit*height - top_crop) % this.state.n_slices;
    var crop_height = bottom_crop - top_crop;

    this.frame_constants = {
      height      : height,
      width       : width,
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
                                .resizeBilinear([this.state.model_height, this.state.model_width])
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
      const slices = mask.mul(tf.equal(mask, tf.onesLike(mask)))
                          .slice(
                            [this.frame_constants.top_crop, 0], 
                            [this.frame_constants.crop_height, this.frame_constants.width]
                          )
                          .reshape(
                            [this.state.n_slices, 
                            this.frame_constants.crop_height/this.state.n_slices, 
                            this.frame_constants.width]
                          )

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

  controller_module_steer = (bearing) => {
    
    console.log('Initial steer: ', bearing);
    
    // experimentally attained estimation to map image bearing to real life bearing
    let real_bearing = 0;
    let poly_deg = this.controller_state.pers_poly.length - 1

    for (const poly_coef of this.controller_state.pers_poly){
      real_bearing += poly_coef * bearing**poly_deg;
      poly_deg -= 1;
    }
    // console.log('After pers transform: ', real_bearing);
    
    let tar_steer = this.controller_state.pid.update(-real_bearing);
    console.log('After pid: ', tar_steer);

    // mapping zero to middle and clip
    tar_steer = this.controller_state.steer_mid + tar_steer * this.controller_state.step_per_degree
    tar_steer = Math.min(tar_steer, this.controller_state.steer_max)
    tar_steer = Math.max(tar_steer, this.controller_state.steer_min)

    return tar_steer;
  }

  componentWillMount() {
    this.virtualVideoElement = this.makeVirtualVideoElement();
  }

  componentDidMount = async () => {
    this.model = await tf.loadGraphModel(`http://127.0.0.1:81/scooter_513/model.json`);
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
    this.initialise_tf_constants(this.state.model_height, this.state.model_width);  // deprecate when fixed frame size is being used
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
    
    // scaling required to map from model dims to canvas dims
    const width_scale = this.props.options.width / this.state.model_width
    const height_scale = this.props.options.height / this.state.model_height

    context.strokeStyle = '#ff0000';

    context.beginPath();
    var i = 0;
    while (coords[i] <= 1) {
      i++;
    }
    context.moveTo(
      coords[i] * width_scale, 
      coords[i+this.state.n_slices] * height_scale
    );

    while (i < this.state.n_slices - 1) {
      if (coords[i+1] > 1) {
        context.lineTo(
          coords[i+1] * width_scale,
          coords[i+this.state.n_slices+1] * height_scale
        );
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
      let steer = -Math.atan(grad) * 180 / Math.PI;
      steer = this.controller_module_steer(steer);
      console.log('Final steer value: ', steer);
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
