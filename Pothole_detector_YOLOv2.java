package ai.certifai.solution.object_detection.pothole_detector;

import ai.certifai.solution.object_detection.Pothole.potholeiterator;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.io.File;
import java.io.IOException;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_DUPLEX;
import static org.bytedeco.opencv.helper.opencv_core.RGB;

public class Pothole_detector_YOLOv2 {

    private static final double learningrate = 0.0001;
    private static final int nboxes = 5;
    private static final int seed = 123;
    private static final int nclasses = 4;
    private static final double lambdanoobj = 0.5;
    private static final double lambdacoord = 5.0;
    private static final int batchsize = 2;
    private static List<String> label;

    private static File modelfilename = new File(System.getProperty("user.dir"), "generated-models/pothole_detector_YOLOv2.zip");
    private static ComputationGraph model;
    private static final double[][] priorboxes = {{1, 3}, {2.5, 5}, {3, 4}, {3.5, 8}, {4, 9}};
    private static final int nepochs = 10;
    private static final double detectionthreshold = 0.4;
    private static final Scalar GREEN = RGB(0, 255.0, 0);
    private static final Scalar YELLOW = RGB(0, 255, 255);
    private static final Scalar BLUE = RGB(0, 0, 255);
    private static final Scalar RED = RGB(255, 0 , 0);
    private static Scalar[] colormap = {GREEN, YELLOW, BLUE, RED};
    private static String labeltext = null;
    private static double eval;


    public static void main(String[] args) throws IOException, InterruptedException {

        potholeiterator.setup();
        RecordReaderDataSetIterator trainiter = potholeiterator.trainiter(batchsize);
        RecordReaderDataSetIterator testiter = potholeiterator.testiter(1);
        label = trainiter.getLabels();

        if (modelfilename.exists()) {
            Nd4j.getRandom().setSeed(seed);
            model = ModelSerializer.restoreComputationGraph(modelfilename);
            System.out.println(model.summary());
            eval = model.score();
            System.out.println(eval);
        } else {
            Nd4j.getRandom().setSeed(seed);
            INDArray priors = Nd4j.create(priorboxes);
            ComputationGraph pretrained = (ComputationGraph) YOLO2.builder().build().initPretrained();

            FineTuneConfiguration fineTuneConfiguration = getfinetuneconf();
            model = getcomputationgraph(pretrained, priors, fineTuneConfiguration);
            System.out.println(model.summary(InputType.convolutional(potholeiterator.yoloheight, potholeiterator.yoloheight, nclasses)));

            UIServer server = UIServer.getInstance();
            StatsStorage storage = new InMemoryStatsStorage();
            server.attach(storage);
            model.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

            for (int i = 0; i < nepochs; i++) {
                trainiter.reset();
                while (trainiter.hasNext()) {
                    model.fit(trainiter.next());
                }
            }
            ModelSerializer.writeModel(model, modelfilename, true);

        }
        OfflineValidationwithDataset(testiter);
    }

    private static ComputationGraph getcomputationgraph(ComputationGraph pretrained, INDArray priors, FineTuneConfiguration finetuneconf){

        return new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(finetuneconf)
                .removeVertexAndConnections("conv2d_23")
                .removeVertexAndConnections("outputs")
                .addLayer("conv2d_23",
                        new ConvolutionLayer.Builder(1, 1)
                                .nIn(1024)
                                .nOut(nboxes * (5 + nclasses))
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.XAVIER)
                                .weightDecay(0.005)
                                .activation(Activation.IDENTITY)
                                .build(), "leaky_re_lu_22")
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .lambdaNoObj(lambdanoobj)
                                .lambdaCoord(lambdacoord)
                                .boundingBoxPriors(priors.castTo(DataType.FLOAT))
                                .build(), "conv2d_23")
                .setOutputs("outputs")
                .build();
    }

    private static FineTuneConfiguration getfinetuneconf(){

        return FineTuneConfiguration.builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder().learningRate(learningrate)
                        .build())
                .l2(0.0001)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();
    }

    private static void OfflineValidationwithDataset(RecordReaderDataSetIterator test) throws InterruptedException {

        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame canvas = new CanvasFrame("Validation Dataset");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
        Mat converted_mat = new Mat();
        Mat converted_mat_big = new Mat();

        while (test.hasNext() && canvas.isVisible()){

            DataSet ds = test.next();
            INDArray features = ds.getFeatures();
            INDArray results = model.outputSingle(features);
            List<DetectedObject> objs = yout.getPredictedObjects(results, detectionthreshold);
            YoloUtils.nms(objs, 0.5);
            Mat mat = imageLoader.asMat(features);
            mat.convertTo(converted_mat, CV_8U, 255, 0);
            int w = mat.cols() * 2;
            int h = mat.rows() * 2;
            resize(converted_mat, converted_mat_big, new Size(w, h));
            converted_mat_big = drawResults(objs, converted_mat_big, w, h);
            canvas.showImage(converter.convert(converted_mat_big));
            canvas.waitKey();
        }

    }

    private static Mat drawResults(List<DetectedObject> objects, Mat mat, int w, int h) {

        for (DetectedObject obj:objects){
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            String labels = label.get(obj.getPredictedClass());
            int x1 = (int) Math.round(w * xy1[0] / potholeiterator.gridwidth);
            int y1 = (int) Math.round(h * xy1[1] / potholeiterator.gridheight);
            int x2 = (int) Math.round(w * xy2[0] / potholeiterator.gridwidth);
            int y2 = (int) Math.round(h * xy2[1] / potholeiterator.gridheight);
            rectangle(mat, new Point(x1, y1), new Point(x2, y2), colormap[obj.getPredictedClass()], 2, 0, 0);
            labeltext = labels + " " + String.format("%.2f", obj.getConfidence() * 100) + "%";
            int[] baseline = {0};
            Size textsize = getTextSize(labeltext, FONT_HERSHEY_DUPLEX, 1, 1, baseline);
            rectangle(mat, new Point(x1 + 2, y2 - 2), new Point(x1 + 2 + textsize.get(0), y2 - 2 + textsize.get(1)), colormap[obj.getPredictedClass()], FILLED, 0, 0 );
            putText(mat, labeltext, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, RGB(0, 0, 0));

        }
        return mat;
    }


}