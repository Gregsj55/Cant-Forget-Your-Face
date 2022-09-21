$(document).ready(function(){

  const video = document.querySelector("#videoElement");
  video.width = 500;
  video.height = 375;
  var canvas = document.querySelector("#canvasElement");
  canvas.width = 500/2
  canvas.height = 375/2
  var ctx = canvas.getContext('2d');
  photo = document.getElementById('imageElement');
  var localMediaStream = null;
  var capflag = false;
  var snapflag = false;
  var verifyflag = false;

  var port = location.protocol + '//' + document.domain + ':' + location.port;
  console.log("port: " + port);
  var socket = io.connect(port);
  socket.on('connect', function() {
    console.log('Connected!');
  });

  console.log(window.location.href);
  if (window.location.href.includes("createAcc")) {
    //console.log("Create acc");
    document.getElementById("capture_btn").onclick = function() {
      console.log("yo");
      capflag = true;
      document.getElementById('verify_btn').disabled = false;
    }
  }
  else if (window.location.href.includes("login")) {
    console.log("Login");
    document.getElementById("snap_btn").onclick = function() {
      console.log("yo");
      snapflag = true;
    }
    document.getElementById("verify_btn").onclick = function() {
      console.log("yo");
      verifyflag = true;
    }

  }



  function emit_frame() {
    if (!localMediaStream) {
      return;
    }
    ctx.drawImage(video, 0, 0, 500/2, 375/2);
    var dataURL = canvas.toDataURL('image/jpeg');

    var username = document.getElementById("username").value;

    socket.emit('image', dataURL, username, capflag, snapflag, verifyflag);

    // var img = new Image();
    // socket.on('response_back', function(image){
    //
    //   console.log("response_back");
    //
    //   img.src = dataURL;
    //   photo.setAttribute('src', image);
    //
    // });
  }

  if(navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video:true }).then(function(stream) {
      video.srcObject = stream;
      localMediaStream = stream;

      setInterval(function () {
        emit_frame();
      }, 100);

    }).catch(function(error) {
      console.log(error);
    });
  }

});
