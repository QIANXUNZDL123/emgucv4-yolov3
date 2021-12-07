using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;

using Emgu.CV.Features2D;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using Emgu.CV.Dnn;
using System.IO;
using System.Runtime.InteropServices;


namespace Emgucv4Demo1
{
    class Program
    {
        static void Main(string[] args)
        {
            //读取DNN Net
            Net Darknet = DnnInvoke.ReadNetFromDarknet(@"F:\\darknet\\build\\darknet\\x64\\weights\\yolov3.cfg", @"F:\\darknet\\build\\darknet\\x64\\weights\\yolov3.weights");
            //读取coco dataset object names
            //string[] ObjectNames = File.ReadAllLines("H:\\opencv_project\\opencvtext\\opencv_tutorial-master\\data\\models\\object_detection_classes_yolov3.txt");
            string[] ObjectNames = File.ReadAllLines(@"F:\darknet\build\darknet\x64\weights\object_detection_classes_yolov3.txt");

            Darknet.SetPreferableBackend(Emgu.CV.Dnn.Backend.Cuda);
            Darknet.SetPreferableTarget(Target.Cuda);

            var image = new Image<Bgr, byte>("H:\\Emgucv_Project\\image\\dog.jpg");


            Mat inputBlob = DnnInvoke.BlobFromImage(image, 1.0 / 255.0, new Size(416, 416), new MCvScalar(0), true, false);

            VectorOfMat output = new VectorOfMat();
            Darknet.SetInput(inputBlob);
            Darknet.Forward(output);

            //新增三个List，包含物件的Rectangle, 物件分数, 物件的index
            List<Rectangle> rects = new List<Rectangle>();
            List<float> scores = new List<float>();
            List<int> objIndexs = new List<int>();

            //取出YOLOv3执行output layer，共有三层
            for (int l = 0; l < output.Size; l++)
            {
                var boxes = output[l];
                int resultRows = boxes.SizeOfDimension[0];
                int resultCols = boxes.SizeOfDimension[1];

                float[] temp = new float[resultRows * resultCols];
                Marshal.Copy(boxes.DataPointer, temp, 0, temp.Length);

                for (int i = 0; i < resultRows; i++)
                {
                    var subMat = new Mat(boxes.Row(i), new Rectangle(5, 0, resultCols - 5, 1));

                    subMat.MinMax(out double[] minValues, out double[] maxValues, out Point[] minPoints, out Point[] maxPoints);

                    //阈值判断
                    if (maxValues[0] > 0.5)
                    {
                        //取出该物件的rectangle
                        int centerX = (int)(temp[i * resultCols + 0] * image.Width);
                        int centerY = (int)(temp[i * resultCols + 1] * image.Height);
                        int width = (int)(temp[i * resultCols + 2] * image.Width);
                        int height = (int)(temp[i * resultCols + 3] * image.Height);
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;
                        Rectangle rect = new Rectangle(left, top, width, height);

                        //将rectangle, score, object index加入List
                        rects.Add(rect);
                        scores.Add((float)maxValues[0]);
                        objIndexs.Add(maxPoints[0].X);
                    }
                }
            }

            var resultImage = image.Clone();
            //进行图像绘制
            for (int i = 0; i < rects.Count; i++)
            {
                resultImage.Draw(rects[i], new Bgr(0, 0, 255), 2);
                var objName = ObjectNames[objIndexs[i]];
                resultImage.Draw(objName, new Point(rects[i].X, rects[i].Y - 10), FontFace.HersheyTriplex, 0.5, new Bgr(255, 100, 100));
                resultImage.Draw(scores[i].ToString(), new Point(rects[i].X, rects[i].Y + 10), FontFace.HersheyTriplex, 0.5, new Bgr(100, 255, 100));
            }

            resultImage.Save("result.jpg");//图像进行保存

            //NMS-非极大抑制
            var selectedObj = DnnInvoke.NMSBoxes(rects.ToArray(), scores.ToArray(), 0.2f, 0.3f);

            //绘制
            for (int i = 0; i < rects.Count; i++)
            {
                if (selectedObj.Contains(i))
                {
                    image.Draw(rects[i], new Bgr(0, 0, 255), 2);
                    var objName = ObjectNames[objIndexs[i]];
                    image.Draw(objName, new Point(rects[i].X, rects[i].Y - 10), FontFace.HersheyTriplex, 0.5, new Bgr(255, 0, 0));
                    image.Draw(scores[i].ToString(), new Point(rects[i].X, rects[i].Y + 10), FontFace.HersheyTriplex, 0.5, new Bgr(0, 255, 0));
                }
            }

            image.Save("NMSresult.jpg");//保存

            CvInvoke.Imshow("n",image);
            CvInvoke.WaitKey(0);


        }
    }
}
