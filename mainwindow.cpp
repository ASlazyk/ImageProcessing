#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>

#include <string>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <math.h>

#include "NormalizationLib.h"
#include "DispLib.h"
#include "histograms.h"
#include "gradient.h"
#include "RegionU16Lib.h"
#include "StringFcLib.h"


#include <tiffio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace boost;
using namespace std;
using namespace boost::filesystem;
using namespace cv;

//=============================================================================
//------------------------------------------------------------------------------------------------------------------------------
//           Out of Class functions
//------------------------------------------------------------------------------------------------------------------------------



void ASmaxMin(Mat Im, double *pmax, double *pmin)
{
    Mat ImD;
    Im.convertTo(ImD, CV_64F);
    int rows = Im.rows;
    int cols = Im.cols;
    int pxls = rows*cols;

    double max = -1.7E308;
    double min = 1.7E308;

    double *pImD = (double*)(ImD.data);

    for(int i = 0; i < pxls; i++)
    {
        double val = *pImD;
        if(val == val) {
            if(max < val)
                max = val;
            if(min > val)
                min = val;
        }
        pImD++;
    }

    *pmax = max;
    *pmin = min;
    ImD.release();
}

cv::Mat ASmask(Mat Im, double threshold)
{

    Mat ImD;
    Im.convertTo(ImD, CV_64F);
    int rows = Im.rows;
    int cols = Im.cols;
    int pxls = rows*cols;

    Mat MaskOut = Mat::zeros(rows, cols, CV_16U);
    double *pImD = (double*)(ImD.data);
    uint16_t *pMaskOut = (uint16_t*)(MaskOut.data);



    for(int i = 0; i < pxls; i++)
    {
        double val = *pImD;
        if(val == val) {            //check for NaN
            if(val > threshold) {
                *pMaskOut = 1;        //write to mask
            } else {
                *pMaskOut = 0;
            }

         }

        pImD++;
        pMaskOut++;
    }

    ImD.release();
    return MaskOut;
}

cv::Mat ASgetGaborKernel(double sigma, double theta, double lambda, double psi, double gamma)
{
    int ks = 65;                //kernel size
    int hks = (ks - 1) / 2;     //half of the kernel size
    theta = theta * CV_PI / 180;
    psi = psi * CV_PI / 180;
    double sigma_x = sigma;
    double sigma_y = sigma/gamma;



    double x_theta;
    double y_theta;

    cv::Mat kernel(ks, ks, CV_64F);

    for(int y = -hks; y <= hks; y++)
    {
        for(int x = -hks; x <= hks; x++)
        {
            x_theta =  x  * cos(theta) + y * sin(theta);
            y_theta = -x  * sin(theta) + y * cos(theta);
            kernel.at<double>(hks+x, hks+y) =
                    (double)exp(-0.5 / pow(sigma_x, 2) * pow(x_theta, 2)
                                -0.5 / pow(sigma_y, 2) * pow(y_theta, 2))
                              * cos(CV_PI * 2 / lambda * x_theta + psi);
        }
    }

    return kernel;
}

cv::Mat ASgetSingleWaveGaborKernel(int ksize, double sig, double the, double lam, double psi, double gam)
{

    //constant for calculating deg to rad
    double dtr = CV_PI / 180;

    Mat GaborKernel = getGaborKernel(Size(ksize, ksize), sig, the*dtr, lam, psi, gam);

    for (int row = 0; row < ksize; row++)
    {
        int x = (ksize - 1) - row - (ksize - 1) / 2;
        for (int col = 0; col < ksize; col++)
        {
            int y = col - (ksize - 1) / 2;
            if (abs(tan(the*dtr)*x -1 * y) / sqrt((tan(the*dtr) * tan(the*dtr) + 1)) > lam / 2)
                GaborKernel.at<double>(row, col) = 0;
        }
    }

    return GaborKernel;
}

cv::Mat ASgetQuadSingleWaveGaborKernel(int ksize, double sig, double the, double lam, double psi, double gam)
{

    Mat KernelT00  = ASgetSingleWaveGaborKernel(ksize, sig,   (0+the), lam, psi, gam);
    Mat KernelT45  = ASgetSingleWaveGaborKernel(ksize, sig,  (45+the), lam, psi, gam);
    Mat KernelT90  = ASgetSingleWaveGaborKernel(ksize, sig,  (90+the), lam, psi, gam);
    Mat KernelT135 = ASgetSingleWaveGaborKernel(ksize, sig, (135+the), lam, psi, gam);

    //auxilliary matrices
    Mat AuxKernel1;
    Mat AuxKernel2;
    Mat QuadKernel;

    ::max(KernelT00, KernelT90, AuxKernel1);
    ::max(KernelT45, KernelT135, AuxKernel2);
    ::max(AuxKernel1, AuxKernel2, QuadKernel);

    return QuadKernel;
}


//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//          constructor Destructor
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//          CLASS FUNCTIONS
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::OpenImageFolder()
{
    path ImageFolder( ui->lineEditImageFolder->text().toStdWString());

    ui->listWidgetImageFiles->clear();
    for (directory_entry& FileToProcess : directory_iterator(ImageFolder))
    {
        regex FilePattern(ui->lineEditRegexImageFile->text().toStdString());
        if (!regex_match(FileToProcess.path().filename().string().c_str(), FilePattern ))
            continue;
        path PathLocal = FileToProcess.path();
        if (!exists(PathLocal))
        {
            ui->textEditOut->append(QString::fromStdString(PathLocal.filename().string() + " File not exists" ));
            break;
        }
        ui->listWidgetImageFiles->addItem(PathLocal.filename().string().c_str());
    }
    if(ui->listWidgetImageFiles->count() && ui->checkBoxAutoProcessFirstFile->checkState())
        ui->listWidgetImageFiles->setCurrentRow(0);
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ReadImage()
{
    if(ui->checkBoxAutocleanOut->checkState())
        ui->textEditOut->clear();
    int flags;
    //if(ui->checkBoxLoadAnydepth->checkState())
        flags = IMREAD_ANYDEPTH;
    //else
    //    flags = IMREAD_COLOR;
    //flags = IMREAD_COLOR;
    ImIn = imread(FileToOpen.string(), flags);
    if(ImIn.empty())
    {
        ui->textEditOut->append("improper file");
        return;
    }
    string extension = FileToOpen.extension().string();

    if((extension == ".tif" || extension == ".tiff") && ui->checkBoxShowTiffInfo->checkState())
        ui->textEditOut->append(QString::fromStdString(TiffFilePropetiesAsText(FileToOpen.string())));

    if(ui->checkBoxShowMatInfo->checkState())
        ui->textEditOut->append(QString::fromStdString(MatPropetiesAsText(ImIn)));
    //ProcessImages();
}
//------------------------------------------------------------------------------------------------------------------------------

void MainWindow::ShowsScaledImage(Mat Im, string ImWindowName, double displayScale)
{
    if(Im.empty())
    {
        ui->textEditOut->append("Empty Image to show");
        return;
    }
    Mat ImToShow;

    ImToShow = Im.clone();

    if (displayScale != 1.0)
        cv::resize(ImToShow,ImToShow,Size(), displayScale, displayScale, INTER_AREA);
    imshow(ImWindowName, ImToShow);
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ShowImages()
{
    if(ImIn.empty())
        return;

    double scale = 1 / ui->doubleSpinBoxImageScale->value();

    if(ui->checkBoxShowInput->checkState())
    {
        ShowsScaledImage(ImIn, "Input Image", scale);
    }
    else
    {
        destroyWindow("Input Image");
    }
/*
    Mat ImToShow;
    ShowSolidRegionOnImage(LesionMask * 3, ImIn).copyTo(ImToShow);;
    rectangle(ImToShow, Rect(tilePositionX,tilePositionY, tileSizeX, tileSizeY), Scalar(0.0, 255.0, 0.0, 0.0),
              10);
    ui->widgetImageWhole->paintBitmap(ImToShow);
    ui->widgetImageWhole->repaint();
*/
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ProcessImages()
{
    double max;
    double min;
    double *pmax = &max;
    double *pmin = &min;
    ::ASmaxMin(ImIn, pmax, pmin);
    Mat qwe;
    qwe = ::ASmask(ImIn, 5);
    qwe = 65535 * qwe;
    imshow("obraz.png", qwe);
    ui->textEditOut->append("Max : " + QString::number(max));
    ui->textEditOut->append("Min : " + QString::number(min));

}
Mat rotate(Mat src, double angle)   //rotate function returning mat object with parametres imagefile and angle
{
    Mat dst;      //Mat object for output image file
    Point2f pt(src.cols/2., src.rows/2.);          //point from where to rotate
    Mat r = getRotationMatrix2D(pt, angle, 1.0);      //Mat object for storing after rotation
    warpAffine(src, dst, r, Size(src.cols, src.rows));  ///applie an affine transforation to image.
    return dst;         //returning Mat object for output image file
}


void MainWindow::ShowKernel()
{
    Mat ImpulseResponse = Mat::zeros(101, 101, CV_16U);
    ImpulseResponse.at<uint16>(50, 50) = 32000;

    double sig = ui->doubleSpinBoxSigma->value();
    double the = ui->doubleSpinBoxTheta->value();
    double lam = ui->doubleSpinBoxLambda->value();
    double psi = ui->doubleSpinBoxPsi->value();
    double gam = ui->doubleSpinBoxGamma->value();
    int ksize = ui->spinBoxKernelSize->value();


    //constant for calculating deg to rad
    double dtr = CV_PI / 180;

    //Mat GaborKernel = getGaborKernel(Size(ksize, ksize), sig, the*dtr, lam, psi, gam);
    Mat GaborKernel = ASgetSingleWaveGaborKernel(ksize, sig, the, lam, psi, gam);



    filter2D(ImpulseResponse, ImpulseResponse, CV_16U, GaborKernel);

    double max = 65535;
    double min = 0;
    double *pmax = &max;
    double *pmin = &min;

    NormParamsMinMax(ImpulseResponse, pmin, pmax);

    Mat GaborMergedDisplay;
    hconcat((ShowImage16Gray(ImpulseResponse, 0, 65000)), ShowImage16PseudoColor(ImpulseResponse, 0, 65000), GaborMergedDisplay);


    //8 kernels for 4 different angles and 2 lambda values
    Mat GaborKernelL06  = ASgetQuadSingleWaveGaborKernel(17, 5, the,  6, 0.5, 0);
    Mat GaborKernelL16  = ASgetQuadSingleWaveGaborKernel(17, 5, the, 16, 0.5, 0);


    //calculating impulse response
    Mat ImpulseResponseL06 = Mat::zeros(101, 101, CV_16U);
    ImpulseResponseL06.at<uint16>(50, 50) = 32000;
    filter2D(ImpulseResponseL06, ImpulseResponseL06, CV_16U, GaborKernelL06);

    Mat ImpulseResponseL16 = Mat::zeros(101, 101, CV_16U);
    ImpulseResponseL16.at<uint16>(50, 50) = 32000;
    filter2D(ImpulseResponseL16, ImpulseResponseL16, CV_16U, GaborKernelL16);


    //auxiliary matrix
    Mat GaborMerged06and16;

    //concatenation of matrices into single image to be shown
    hconcat((ShowImage16PseudoColor(ImpulseResponseL06, 0, 65000)), ShowImage16PseudoColor(ImpulseResponseL16, 0, 65000), GaborMerged06and16);
    vconcat(GaborMergedDisplay, GaborMerged06and16, GaborMergedDisplay);

    //showing image
    ShowsScaledImage(GaborMergedDisplay, "Impuls Response", ui->doubleSpinBoxImageScale->value());


}

//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//          Slots
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------

void MainWindow::on_pushButtonOpenImageFolder_clicked()
{
    QFileDialog dialog(this, "Open Folder");
    //dialog.setFileMode(QFileDialog::Directory);
    //dialog.setDirectory(ui->lineEditImageFolder->text());
    path ImageFolder;
    if(dialog.exec())
    {
        ImageFolder = dialog.directory().path().toStdWString();
    }
    else
        return;
    if (!exists(ImageFolder))
    {
        ui->textEditOut->append(" Image folder : " + QString::fromStdWString(ImageFolder.wstring())+ " not exists ");
        return;
    }
    if (!is_directory(ImageFolder))
    {
        ui->textEditOut->append(" Image folder : " + QString::fromStdWString(ImageFolder.wstring())+ " This is not a directory path ");
        return;
    }
    ui->lineEditImageFolder->setText(QString::fromStdWString(ImageFolder.wstring()));

    OpenImageFolder();
}

void MainWindow::on_listWidgetImageFiles_currentRowChanged(int currentRow)
{
    FileToOpen = ui->lineEditImageFolder->text().toStdWString();
    FileToOpen.append(ui->listWidgetImageFiles->item(currentRow)->text().toStdWString());
    if(!exists(FileToOpen))
    {
        ui->textEditOut->append(" file : " + QString::fromStdWString(FileToOpen.wstring())+ " not exists ");
        return;
    }

    ReadImage();
    ProcessImages();
    ShowImages();
}

void MainWindow::on_pushButtonProcessAllFiles_clicked()
{
    int filesCount = ui->listWidgetImageFiles->count();
    int firstFile = 0;

    ui->textEditOut->append("jeden");

    Mat ImSum;
    Mat ImMask;
    Mat ImAvg;
    int isImSumCreatedFlag = 0;

    double *pImSum = (double*)(ImSum.data);
    double *pImAvg = (double*)(ImAvg.data);

    ui->textEditOut->clear();
    //time_t begin,end;
    //time (&begin);
    for(int fileNr = firstFile; fileNr< filesCount; fileNr++)
    {
        FileToOpen = ui->lineEditImageFolder->text().toStdWString();
        FileToOpen.append(ui->listWidgetImageFiles->item(fileNr)->text().toStdWString());
        if(!exists(FileToOpen))
        {
            ui->textEditOut->append(" file : " + QString::fromStdWString(FileToOpen.wstring())+ " not exists ");
            return;
        }

        ReadImage();

        if(isImSumCreatedFlag == 0) {
            ImSum = Mat::zeros(ImIn.rows, ImIn.cols, CV_64F);
            int isImSumCreatedFlag = 0;
            isImSumCreatedFlag = 1;
        }

        //ProcessImages();
        //ShowImages();

        ImMask = ::ASmask(ImIn, 5);
        int *pImIn = (int*)(ImIn.data);
        double *pImMask = (double*)(ImMask.data);
        int ImPxls = ImMask.rows * ImMask.cols;

        for(int i = 0; i < ImPxls; i++) {
            if(pImMask != 0)
               *pImSum += *pImIn;
            pImSum++;
            pImIn++;
            pImMask++;
        }

        imshow("qwe", ImSum);

        waitKey(200);

    }

    for (int i = 0; i < (ImSum.rows * ImSum.cols); i++) {
        *pImAvg = *pImSum / filesCount;
        pImSum++;
        pImAvg++;
    }

    //time (&end);
    //double difference = difftime (end,begin);
    //QString TimeStringQ = " calcTime = " + QString::number(difference) + " s" + "\n";
    //ui->textEditOut->append(TimeStringQ);
}

void MainWindow::on_checkBoxShowInput_toggled(bool checked)
{
    ShowImages();
}

void MainWindow::on_doubleSpinBoxImageScale_valueChanged(double arg1)
{
    ShowImages();
}

void MainWindow::on_pushButtonShowKernel_clicked()
{
    ShowKernel();
}


void MainWindow::on_doubleSpinBoxSigma_valueChanged(double arg1)
{
    ShowKernel();
}


void MainWindow::on_doubleSpinBoxTheta_valueChanged(double arg1)
{
    ShowKernel();
}


void MainWindow::on_doubleSpinBoxLambda_valueChanged(double arg1)
{
    ShowKernel();
}


void MainWindow::on_doubleSpinBoxGamma_valueChanged(double arg1)
{
    ShowKernel();
}


void MainWindow::on_doubleSpinBoxPsi_valueChanged(double arg1)
{
    ShowKernel();
}


void MainWindow::on_spinBoxKernelSize_valueChanged(int arg1)
{
    ShowKernel();
}

