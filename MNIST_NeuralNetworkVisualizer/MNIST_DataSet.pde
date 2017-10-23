import java.io.FileInputStream;
import java.io.IOException;

class MNIST_DataSet {
    private String path;
    private static final String trainImageFile = "train-images.idx3-ubyte";
    private static final String trainLabelFile = "train-labels.idx1-ubyte";
    private static final String testImageFile = "t10k-images.idx3-ubyte";
    private static final String testLabelFile = "t10k-labels.idx1-ubyte";
    public static final int trainAmount = 60_000;
    public static final int testAmount  = 10_000;
    public static final int pixelsPerImage = 784;
    public static final int labelAmount =     10;
    private byte[] pixels;
    private double[][][] trainPairs;
    private double[][][] testPairs;
    private int[][][] trainPairsRaw;
    private int[][][] testPairsRaw;

    public MNIST_DataSet(String path) throws IOException {
        this.path = path;
        pixels = new byte[pixelsPerImage];
        loadTrainPairs();
        loadTestPairs();
    }

    public double[][] getTrainPair(int index) {
        return trainPairs[index];
    }

    public double[][] getTestPair(int index) {
        return testPairs[index];
    }
    
    public int[][] getTrainPairRaw(int index) {
        return trainPairsRaw[index];
    }

    public int[][] getTestPairRaw(int index) {
        return testPairsRaw[index];
    }

    private void loadTrainPairs() throws IOException {
        FileInputStream imageReader = new FileInputStream(path+ trainImageFile);
        FileInputStream labelReader = new FileInputStream(path+ trainLabelFile);
        imageReader.skip(16);
        labelReader.skip(8);
        trainPairs = new double[trainAmount][2][];
        trainPairsRaw = new int[trainAmount][2][];
        for(int i = 0; i < trainAmount; i++) {
            trainPairs[i][0] = new double[pixelsPerImage];
            trainPairsRaw[i][0] = new int[pixelsPerImage];
            imageReader.read(pixels);
            for(int j = 0; j < pixelsPerImage; j++){
                trainPairsRaw[i][0][j] = (int)(pixels[j] & 0xff);
                trainPairs[i][0][j] = (trainPairsRaw[i][0][j])* 1./255;
            }
                
            trainPairs[i][1] = new double[labelAmount];
            trainPairsRaw[i][1] = new int[] {labelReader.read()};
            trainPairs[i][1][trainPairsRaw[i][1][0]] = 1;
        }
        imageReader.close();
        labelReader.close();
    }
    
    private void loadTestPairs() throws IOException {
        FileInputStream imageReader = new FileInputStream(path+ testImageFile);
        FileInputStream labelReader = new FileInputStream(path+ testLabelFile);
        imageReader.skip(16);
        labelReader.skip(8);
        testPairs = new double[testAmount][2][];
        testPairsRaw = new int[testAmount][2][];
        for(int i = 0; i < testAmount; i++) {
            testPairs[i][0] = new double[pixelsPerImage];
            testPairsRaw[i][0] = new int[pixelsPerImage];
            imageReader.read(pixels);
            for(int j = 0; j < pixelsPerImage; j++){
                testPairsRaw[i][0][j] = (int)(pixels[j] & 0xff);
                testPairs[i][0][j] = (testPairsRaw[i][0][j])* 1./255;
            }
                
            testPairs[i][1] = new double[labelAmount];
            testPairsRaw[i][1] = new int[] {labelReader.read()};
            testPairs[i][1][testPairsRaw[i][1][0]] = 1;
        }
        imageReader.close();
        labelReader.close();
    }
}