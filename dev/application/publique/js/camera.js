document.addEventListener('DOMContentLoaded', function() {
    const videoElement = document.getElementById('barcode-scanner'); 

    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
      .then(function(stream) {
        videoElement.srcObject = stream;
        videoElement.play();

        Quagga.init({
          inputStream: {
            name: "Live",
            type: "LiveStream",
            target: videoElement, 
            constraints: {
              facingMode: "environment"
            },
          },
          decoder: {
            readers: ['ean_reader']
          }
        }, function(err) {
          if (err) {
            console.error(err);
            return;
          }
          Quagga.start();
        });

        Quagga.onDetected(function(barcodeScanner) {
          Quagga.stop();
          stream.getTracks().forEach(track => track.stop());

          videoElement.innerHTML = `Code-barres détecté : ${barcodeScanner.codeResult.code}`;

          const openFoodFactsApiUrl = `https://world.openfoodfacts.org/api/v0/product/${barcodeScanner.codeResult.code}.json`;

          fetch(openFoodFactsApiUrl)
            .then(response => response.json())
            .then(data => {
              console.log('Données Open Food Facts :', data);
            })
            .catch(error => {
              console.error('Erreur lors de la requête à Open Food Facts :', error);
            });
        });
      })
      .catch(function(error) {
        console.error('Erreur lors de l\'accès à la caméra :', error);
      });
});
