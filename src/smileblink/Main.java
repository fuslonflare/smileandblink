package smileblink;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;
import org.opencv.core.Core;
import smileblink.controller.Controller;

public class Main extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {
        FXMLLoader loader = new FXMLLoader(getClass().getResource("view/sample.fxml"));
        BorderPane root = loader.load();

        primaryStage.setTitle("Smile and Blink");
        primaryStage.setScene(new Scene(root, 800, 600));
        primaryStage.show();

        // init the controller
        Controller controller = loader.getController();
        controller.init();
    }


    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        launch(args);
    }
}
