

int nodeSize = 25;

PVector[][] neuronPos;
PVector imagePos;
int imageSize;
void setupVisualizer(int[] layers) {

  neuronPos = new PVector[layers.length][];
  float layerDist = width / layers.length;
  imagePos = new PVector(0.5*layerDist, height /2);
  imageSize = (int)layerDist/2;
  neuronPos[0] = new PVector[] {imagePos};
  
  for (int l = 1; l < neuronPos.length; l++) {
    neuronPos[l] = new PVector[layers[l]];
    float x = 0.5*layerDist + l*layerDist;
    float neuronDist = (height-80) / layers[l];
    for (int n = 0; n < neuronPos[l].length; n++) {
      float y = (0.5*neuronDist + n*neuronDist)+20;
      neuronPos[l][n] = new PVector(x, y);
    }
  }
}

void drawVisualizer() {
  tested = false;

  drawNeuralNetwork();

  fill(255);
  textFont(normal);
  textSize(20);
  text(String.format("Set: %5s  Epoch: %5s  Accuracy: %5s/%-5s = %7s  EpochTime: %.3f ms.", 
    set, epoch, correct, MNIST_DataSet.testAmount, 
    String.format("%.2f", 100.*correct/MNIST_DataSet.testAmount)+" %",
    epochTime), 20, height-20);
}



void drawNeuralNetwork() {
  stroke(255);
  strokeWeight(1);
  for(int nj = 0; nj < network.neurons[1].length; nj++)
    line(imagePos.x+imageSize/2, imagePos.y, neuronPos[1][nj].x, neuronPos[1][nj].y);
  
  for (int lj = 1; lj < neuronPos.length; lj++) {
    int lk = lj+1;
    for (int nj = 0; nj < neuronPos[lj].length; nj++) {
      if (lk < neuronPos.length) {
        for (int nk = 0; nk < neuronPos[lk].length; nk++) {
          if (network.w[lj][nj][nk] >= 0) stroke(0, 255, 0);
          else stroke(255, 0, 0);
          strokeWeight((float)sigmoid(Math.abs(network.w[lj][nj][nk]), 2, -0.2, 5));
          line(neuronPos[lj][nj].x, neuronPos[lj][nj].y, neuronPos[lk][nk].x, neuronPos[lk][nk].y);
        }
      }
      
      fill(255);
      textFont(bold);
      textSize(12);
      StringBuilder val = new StringBuilder();
      
      if (lj == neuronPos.length-1) {
        text(String.format("Err: %6s\nVal:  %.3f%s", 
          String.format("%.3f", network.err[lj][nj]), 
          network.neurons[lj][nj].value, 
          val.toString()), 
          neuronPos[lj][nj].x-(val.length()*5), 
          neuronPos[lj][nj].y-nodeSize-12);
      }

      if (network.neurons[lj][nj].bias >= 0) stroke(0, 255, 0);
      else stroke(255, 0, 0);
      strokeWeight((float)sigmoid(Math.abs(network.neurons[lj][nj].bias), 2, -0.2, 5));
      
      fill((int)(network.neurons[lj][nj].value*255));
      ellipse(neuronPos[lj][nj].x, neuronPos[lj][nj].y, nodeSize, nodeSize);

      // Expected output
      if (lj == neuronPos.length -1) {
        stroke(255);
        strokeWeight(1);
        if(isTraining)
          fill((int)(dataSet.getTrainPair(set % MNIST_DataSet.trainAmount)[1][nj])*255); // Because the parallel training thread can for a few cpu cycles make 'set = trainOutput.length' in the for loops set++
        else
          fill((int)(dataSet.getTestPair(set % MNIST_DataSet.testAmount)[1][nj])*255);
        ellipse(neuronPos[lj][nj].x+nodeSize*3, neuronPos[lj][nj].y, nodeSize, nodeSize);
        fill(255); textFont(normal); textSize(20);
        text(nj, neuronPos[lj][nj].x+nodeSize*4, neuronPos[lj][nj].y+nodeSize*0.25);
        if(nj == neuronPos[lj].length-1){
          text("A", neuronPos[lj][nj].x, neuronPos[lj][nj].y+nodeSize*2);
          text("P", neuronPos[lj][nj].x+nodeSize*3, neuronPos[lj][nj].y+nodeSize*2);
        }
      }
    }
  }
}

static double sigmoid(double x, double a, double b, double k) {
  return k / (1 + Math.exp(a+b*x));
}