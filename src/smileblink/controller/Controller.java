package smileblink.controller;

import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import java.io.ByteArrayInputStream;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class Controller {
    @FXML
    private Button btnCamera;
    @FXML
    private ImageView ivOriginalFrame;
    @FXML
    private CheckBox cbFaceClassifier;
    @FXML
    private CheckBox cbSmileClassifier;
    @FXML
    private CheckBox cbBoth;
    @FXML
    private CheckBox cbEyeClassifier;

    private ScheduledExecutorService timer;
    private VideoCapture capture;
    private boolean cameraActive;

    private CascadeClassifier faceCascade;
    private CascadeClassifier smileCascade;
    private CascadeClassifier eyeCascade;
    private int absoluteFaceSize;

    private String pathFaceCascade;
    private String pathSmileCascade;
    private String pathEyeCascade;

    private int currentMode;
    private static final int MODE_FACE_DETECTION = 2;
    private static final int MODE_SMILING_DETECTION = 3;
    private static final int MODE_EYE_DETECTION = 5;

    private int timeSmile;

    public void init() {
        cameraActive = false;
        capture = new VideoCapture();

        faceCascade = new CascadeClassifier();
        smileCascade = new CascadeClassifier();
        eyeCascade = new CascadeClassifier();
        absoluteFaceSize = 0;

        String pathCascade = "resources/haarcascades/";
        pathFaceCascade = pathCascade + "haarcascade_frontalface_alt2.xml";
        pathSmileCascade = pathCascade + "haarcascade_smile.xml";
        pathEyeCascade = pathCascade + "haarcascade_eye.xml";
        timeSmile = 0;
    }

    @FXML
    protected void startCamera() {
        if (!cbEyeClassifier.isSelected() && !cbBoth.isSelected()
                && !cbFaceClassifier.isSelected() && !cbSmileClassifier.isSelected()) {
            Alert alert = new Alert(Alert.AlertType.ERROR);
            alert.setTitle("Error Alert!");
            alert.setHeaderText(null);
            alert.setContentText("กรุณากลับไปเลือกโหมดที่ต้องการ");

            alert.showAndWait();
            return;
        }

        ivOriginalFrame.setFitWidth(600);
        ivOriginalFrame.setPreserveRatio(true);

        if (!cameraActive) {
            cbSmileClassifier.setDisable(true);
            cbFaceClassifier.setDisable(true);
            cbBoth.setDisable(true);
            cbEyeClassifier.setDisable(true);

            capture.open(1);

            if (capture.isOpened()) {
                cameraActive = true;

                if (cbSmileClassifier.isSelected()) {
                    currentMode = MODE_SMILING_DETECTION;
                } else if (cbFaceClassifier.isSelected()) {
                    currentMode = MODE_FACE_DETECTION;
                } else if (cbEyeClassifier.isSelected()) {
                    currentMode = MODE_EYE_DETECTION * MODE_FACE_DETECTION;
                } else if (cbBoth.isSelected()) {
                    currentMode = MODE_FACE_DETECTION * MODE_SMILING_DETECTION;
                }

                // Grab a frame every 33 millisec. (33 ms = 1 frame
                //                                  1000 ms (1 sec) = 1*1000/33 -> 30)
                // 30 frames per second
                Runnable frameGrabber = () -> {
                    Image imageToShow = grabFrame();
                    ivOriginalFrame.setImage(imageToShow);
                };

                timer = Executors.newSingleThreadScheduledExecutor();
                timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

                btnCamera.setText("Stop Camera");
            } else {
                System.err.println("Camera connection failed");
            }
        } else {
            cameraActive = false;
            btnCamera.setText("Start Camera");

            cbFaceClassifier.setDisable(false);
            cbSmileClassifier.setDisable(false);
            cbBoth.setDisable(false);
            cbEyeClassifier.setDisable(false);

            // Stop timer
            timer.shutdown();
            try {
                timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                System.err.println("Exception from trying to stop frame capture, " +
                        "try to release camera. /" + e);
            }

            capture.release();
            ivOriginalFrame.setImage(null);
        }
    }

    private Image grabFrame() {
        Image imageToShow = null;
        Mat frame = new Mat();

        if (capture.isOpened()) {
            try {
                capture.read(frame);
                if (!frame.empty()) {
                    detectAndDisplay(frame);
                    imageToShow = matToImage(frame);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return imageToShow;
    }

    private void detectAndDisplay(Mat frame) {
        Mat grayFrame = new Mat();

        Rect[] smilesArray;
        Rect[] facesArray;
        Rect[] eyeArray;

        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(grayFrame, grayFrame);

        if (absoluteFaceSize == 0) {
            int height = grayFrame.rows();
            if (Math.round(height * 0.2f) > 0) {
                absoluteFaceSize = Math.round(height * 0.2f);
            }
        }

        Point latestBrFace = new Point();
        Point latestTlFace = new Point();

        if (currentMode % MODE_FACE_DETECTION == 0) {
            MatOfRect faces = new MatOfRect();
            faceCascade.detectMultiScale(
                    grayFrame,
                    faces,
                    1.1,
                    2,
                    Objdetect.CASCADE_SCALE_IMAGE,
                    new Size(absoluteFaceSize, absoluteFaceSize),
                    new Size()
            );
            facesArray = faces.toArray();
            for (int i = 0; i < facesArray.length; i++) {
                Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(34.0, 87.0, 255.0), 3);
                latestBrFace = facesArray[i].br();
                latestTlFace = facesArray[i].tl();
            }
        }
        if (currentMode % MODE_SMILING_DETECTION == 0) {
            MatOfRect smiles = new MatOfRect();
            smileCascade.detectMultiScale(
                    grayFrame,
                    smiles,
                    1.1,
                    2,
                    Objdetect.CASCADE_SCALE_IMAGE,
                    new Size(absoluteFaceSize, absoluteFaceSize),
                    new Size()
            );
            smilesArray = smiles.toArray();
            for (int i = 0; i < smilesArray.length; i++) {
                if (currentMode == MODE_FACE_DETECTION * MODE_SMILING_DETECTION) {
                    if (smilesArray[i].tl().x <= latestBrFace.x + 10 &&
                            smilesArray[i].tl().x >= latestTlFace.x - 10 &&
                            smilesArray[i].tl().y >= (latestTlFace.y + latestBrFace.y) / 2.0 &&
                            smilesArray[i].br().y <= latestBrFace.y + 10 &&
                            smilesArray[i].br().x <= latestBrFace.x + 10 &&
                            smilesArray[i].br().x >= latestTlFace.x - 10) {
                        // TODO: Set Trigger when kids smile here!
                        Imgproc.rectangle(frame, smilesArray[i].tl(), smilesArray[i].br(), new Scalar(99.0, 30.0, 233.0), 3);
                    }
                } else {
                    Imgproc.rectangle(frame, smilesArray[i].tl(), smilesArray[i].br(), new Scalar(99.0, 30.0, 233.0), 3);
                }
            }
        }
        if (currentMode % MODE_EYE_DETECTION == 0) {
            MatOfRect eyes = new MatOfRect();
            eyeCascade.detectMultiScale(
                    grayFrame,
                    eyes,
                    1.1,
                    2,
                    Objdetect.CASCADE_SCALE_IMAGE,
                    new Size(absoluteFaceSize, absoluteFaceSize),
                    new Size()
            );
            eyeArray = eyes.toArray();
            for (int i = 0; i < eyeArray.length; i++) {
                if (currentMode == MODE_EYE_DETECTION * MODE_FACE_DETECTION) {
                    if (eyeArray[i].tl().x <= latestBrFace.x &&
                            eyeArray[i].tl().x >= latestTlFace.x &&
                            eyeArray[i].tl().y <= (latestTlFace.y + latestBrFace.y) / 2.0 &&
                            eyeArray[i].br().x <= latestBrFace.x &&
                            eyeArray[i].br().x >= latestTlFace.x) {
                        Imgproc.rectangle(frame, eyeArray[i].tl(), eyeArray[i].br(), new Scalar(99.0, 30.0, 233.0), 3);
                        if (eyeArray.length == 2) {
                            // TODO: Change background when kids blink here!
                            log("blink");
                        }
                    }
                }
            }
        }
    }

    @FXML
    protected void selectFace() {
        if (cbSmileClassifier.isSelected()) {
            cbSmileClassifier.setSelected(false);
        }
        if (cbBoth.isSelected()) {
            cbBoth.setSelected(false);
        }
        if (cbEyeClassifier.isSelected()) {
            cbEyeClassifier.setSelected(false);
        }
        faceCascade.load(pathFaceCascade);
        btnCamera.setDisable(false);
    }

    @FXML
    protected void selectSmile() {
        if (cbFaceClassifier.isSelected()) {
            cbFaceClassifier.setSelected(false);
        }
        if (cbBoth.isSelected()) {
            cbBoth.setSelected(false);
        }
        if (cbEyeClassifier.isSelected()) {
            cbEyeClassifier.setSelected(false);
        }
        smileCascade.load(pathSmileCascade);
        btnCamera.setDisable(false);
    }

    @FXML
    protected void selectBoth() {
        if (cbFaceClassifier.isSelected()) {
            cbFaceClassifier.setSelected(false);
        }
        if (cbSmileClassifier.isSelected()) {
            cbSmileClassifier.setSelected(false);
        }
        if (cbEyeClassifier.isSelected()) {
            cbEyeClassifier.setSelected(false);
        }
        smileCascade.load(pathSmileCascade);
        faceCascade.load(pathFaceCascade);
        btnCamera.setDisable(false);
    }

    @FXML
    protected void selectEye() {
        if (cbFaceClassifier.isSelected()) {
            cbFaceClassifier.setSelected(false);
        }
        if (cbSmileClassifier.isSelected()) {
            cbSmileClassifier.setSelected(false);
        }
        if (cbBoth.isSelected()) {
            cbBoth.setSelected(false);
        }
        eyeCascade.load(pathEyeCascade);
        faceCascade.load(pathFaceCascade);
        btnCamera.setDisable(false);
    }

    /**
     * Convert a Mat object (OpenCV) in the corresponding Image for JavaFX
     *
     * @param frame the {@link Mat} representing the current frame
     * @return the {@link Image} to show
     */
    private Image matToImage(Mat frame) {
        // create a temporary buffer
        MatOfByte buffer = new MatOfByte();
        // encode the frame in the buffer, according to the PNG format
        Imgcodecs.imencode(".png", frame, buffer);
        // build and return an Image created from the image encoded in the
        // buffer
        return new Image(new ByteArrayInputStream(buffer.toArray()));
    }

    private void log(String text) {
        System.out.println(String.valueOf(text));
    }
}
