package test;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class MNISTLoader {
    public static double[][] loadImages(String filepath) throws IOException {
        DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filepath)));

        int magicNumber = dis.readInt(); // Should be 2051
        int numberOfImages = dis.readInt();
        int rows = dis.readInt();
        int columns = dis.readInt();
        int imageSize = rows * columns;

        double[][] images = new double[numberOfImages][imageSize];
        byte[] imageBuffer = new byte[imageSize]; // Temporary buffer for one image

        for (int i = 0; i < numberOfImages; i++) {
            // Read the entire image into the buffer in one go
            dis.readFully(imageBuffer);

            for (int j = 0; j < rows * columns; j++) {
                // Convert signed byte to unsigned and normalise
                // (imageBuffer[j] & 0xFF) converts -128...127 to 0...255
                images[i][j] = (imageBuffer[j] & 0xFF) / 255.0;
            }
        }
        dis.close();
        return images;
    }

    public static int[] loadLabels(String filepath) throws IOException {
        DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filepath)));

        int magicNumber = dis.readInt(); // Should be 2049
        int numberOfLabels = dis.readInt();

        int[] labels = new int[numberOfLabels];
        byte[] labelBuffer = new byte[numberOfLabels];

        dis.readFully(labelBuffer);

        for (int i = 0; i < numberOfLabels; i++) {
            // Convert to unsigned int
            labels[i] = labelBuffer[i] & 0xFF;
        }
        dis.close();
        return labels;
    }

//    public static void main(String[] args) throws IOException {
//        double[][] trainImages = loadImages("train-images.idx3-ubyte");
//        int[] trainLabelsRaw = loadLabels("train-labels.idx1-ubyte");
//
//        double[][] trainLabels = MNISTClassification.oneHotEncode(trainLabelsRaw);
//
//        System.out.println(trainImages.length + ", " + trainImages[0].length);
//        System.out.println(trainLabelsRaw.length);
//        System.out.println(trainLabels.length + ", " + trainLabels[0].length);
//    }
}
