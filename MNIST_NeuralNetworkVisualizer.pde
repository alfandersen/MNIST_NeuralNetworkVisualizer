import java.io.FileInputStream;
import java.io.IOException;

MNIST_DataSet dataSet;
NeuralNetwork network;

int[]layers = new int[] {MNIST_DataSet.pixelsPerImage, 16,16, MNIST_DataSet.labelAmount};



PImage image;
PImage num;
int val = -1;

int epoch = 0;
int set = 0;
int correct = 0;

PFont bold;
PFont normal;

boolean isTraining = false;
double epochTime;

boolean tested = false;
int[] trainPredictions;
int[] testPredictions;
boolean fileNotFound = false;
void setup(){
  fullScreen();
  
  bold = createFont("VeraMono-Bold.ttf", 20);
  normal = createFont("VeraMono.ttf", 20);
  textFont(normal);
  

  trainPredictions = new int[MNIST_DataSet.trainAmount];
  testPredictions = new int[MNIST_DataSet.testAmount];
  
  
  image = createImage(28, 28, RGB);
  num = createImage(width,height,RGB);
  fill(255);
  textFont(normal);
  textSize(20);
  try{
    dataSet = new MNIST_DataSet(sketchPath("data\\"));
    network = new NeuralNetwork(layers, 1L);
    setupVisualizer(layers);
    test();
  }
  catch(IOException e){
    fileNotFound = true;
    e.printStackTrace();
    println("Dataset not found.\nDownload the four files from http://yann.lecun.com/exdb/mnist/ and put them in the /data folder");
    background(0);
    text("Dataset not found.\nDownload the four files from http://yann.lecun.com/exdb/mnist/ and put them in the /data folder", 20, height/2);
    return;
    //stop();
  }
  imageMode(CENTER);
}

void draw(){
  if(fileNotFound) return;
  background(20);
  updateImage();
  drawVisualizer();
  image(image,300,height/2);
  
  int prediction;
  int actual;
  if(isTraining){ 
    prediction = trainPredictions[set == 0 ? 0 : set-1];
    actual = dataSet.getTrainPairRaw(set == 0 ? 0 : set-1)[1][0];
  } else {
    prediction = testPredictions[set];
    actual = dataSet.getTestPairRaw(set)[1][0];
  }
  textSize(50);
  fill(prediction == actual ? color(0,255,0) : color(255,0,0));
  text(String.format("Predicted:%2d\nActual:%5d", prediction, actual), 100, height/2+200);
}

void updateImage(){
  int[][] pairRaw;
  
  if(isTraining)
    pairRaw = dataSet.getTrainPairRaw(set);
  else 
    pairRaw = dataSet.getTestPairRaw(set);
    
  image.resize(28,28);
  for(int i = 0; i < MNIST_DataSet.pixelsPerImage; i++)
    image.pixels[i] = color(pairRaw[0][i]);
    
  image.updatePixels();
  image.resize(imageSize,imageSize);
  val = pairRaw[1][0];
}

void keyPressed() {
  if(fileNotFound) return;
  if (isTraining) isTraining = false;
  else {
    switch(key) {
    case 'q':  trainEpochs(1);  break;
    case 'w':  trainEpochs(10);  break;
    case 'e':  trainEpochs(100);  break;
    case CODED: switch(keyCode){
        case RIGHT: set = (set+1)%MNIST_DataSet.testAmount; break;
        case LEFT:  set = (set+MNIST_DataSet.testAmount-1)%MNIST_DataSet.testAmount; break;
      }
    network.predict(dataSet.getTestPair(set)[0]); break;
    case '0': findNext(0); break;
    case '1': findNext(1); break;
    case '2': findNext(2); break;
    case '3': findNext(3); break;
    case '4': findNext(4); break;
    case '5': findNext(5); break;
    case '6': findNext(6); break;
    case '7': findNext(7); break;
    case '8': findNext(8); break;
    case '9': findNext(9); break;
    case 'x': findNextError(); break;
    default:
      set = (set+1)%MNIST_DataSet.testAmount;
      network.predict(dataSet.getTestPair(set)[0]);
      break;
    }
  }
}

void findNext(int target){
  do 
    set = (set+1)%MNIST_DataSet.testAmount;
  while(dataSet.getTestPairRaw(set)[1][0] != target);
    
  network.predict(dataSet.getTestPair(set)[0]);
}

void findNextError(){
  do {
    set = (set+1)%MNIST_DataSet.testAmount;
    network.predict(dataSet.getTestPair(set)[0]);
  } while(testPredictions[set] == dataSet.getTestPairRaw(set)[1][0]);
}

void test() {
  int correct = 0;
  for (set = 0; set < MNIST_DataSet.testAmount; set++) {
    double[] out = network.predict(dataSet.getTestPair(set)[0]);
    int maxId = 0;
    for (int x = 1; x < MNIST_DataSet.labelAmount; x++) {
      maxId = out[x] > out[maxId] ? x : maxId;
    }
    testPredictions[set] = maxId;
    if (maxId == dataSet.getTestPairRaw(set)[1][0]) correct++;
  }
  this.correct = correct;
  set = 0;
  network.predict(dataSet.getTestPair(set)[0]);
  tested = true;
}

void trainEpochs(final int e) {
  new Thread(new Runnable() {
    public void run() {
      int endEpoch = epoch+e;
      isTraining = true;
      for (; epoch < endEpoch && isTraining; epoch++) {
        network.setLearningRate(1./(1.0001+0.05*epoch)+0.0001);
        long time = System.nanoTime();
        for (set = 0; set < MNIST_DataSet.trainAmount && isTraining; set++) {
          network.train(dataSet.getTrainPair(set)[0], dataSet.getTrainPair(set)[1]);
          int maxId = 0;
          for (int x = 1; x < MNIST_DataSet.labelAmount; x++) {
            maxId = network.neurons[network.outputLayer][x].value > network.neurons[network.outputLayer][maxId].value ? x : maxId;
          }
          trainPredictions[set] = maxId;
        }
        epochTime = 1E-6*(System.nanoTime()-time);
        test();
      }
      set = 0;
      //test();
      isTraining = false;
    }
  }
  ).start();
}