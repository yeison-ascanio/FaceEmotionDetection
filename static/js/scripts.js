document.addEventListener("DOMContentLoaded", function() {
    const preloader = document.getElementById('preloader');
    const videoFeed = document.getElementById('videoFeed');
    const statusMessage = document.getElementById('statusMessage');

    const socket = io(); // Conectar con el servidor SocketIO

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            const mediaStreamTrack = stream.getVideoTracks()[0];
            const imageCapture = new ImageCapture(mediaStreamTrack);
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');

            videoFeed.style.display = 'block';
            statusMessage.style.display = 'none';

            const sendFrames = () => {
                imageCapture.grabFrame()
                    .then(imageBitmap => {
                        canvas.width = imageBitmap.width;
                        canvas.height = imageBitmap.height;
                        context.drawImage(imageBitmap, 0, 0, canvas.width, canvas.height);

                        canvas.toBlob((blob) => {
                            const reader = new FileReader();
                            reader.onload = function() {
                                const base64data = reader.result.split(',')[1];
                                socket.emit('process_frame', { frame: base64data });
                            };
                            reader.readAsDataURL(blob);
                        }, 'image/jpeg');

                        setTimeout(sendFrames, 200); // Envía un frame cada 200ms
                    })
                    .catch(error => {
                        console.error('Error al capturar el frame:', error);
                    });
            };

            sendFrames();

            videoFeed.addEventListener('load', function() {
                preloader.style.display = 'none';
                statusMessage.style.display = 'none';
                videoFeed.style.display = 'block';
            });
        })
        .catch(function(error) {
            statusMessage.textContent = 'Error al conectar con la cámara.';
            statusMessage.style.display = 'block';
            console.error("Error al acceder a la cámara:", error);
            preloader.style.display = 'none';
        });

    socket.on('frame_processed', function(data) {
        const frame = 'data:image/jpeg;base64,' + data.frame;
        videoFeed.src = frame;
    });
});
