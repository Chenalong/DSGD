
import org.apache.spark.Logging

/**
 * Created by Administrator on 2016/6/15 0015.
 */
class DSGD(numFactors: Int, numWorkers: Int, numIterations: Int, beta: Double, lambda: Double, ratingFilePath: String,
           uFilePath: String, vFilePath: String,initStep:Double) extends Serializable with Logging
{
    /**
     *numFactors : The length of user or item hide feature vector
     * numWorkers: The degree of parallelism
     * numIterations: The quanlity of iterations
     * initStep : initial update steps
     * beta : The parameter of steps, following is update formula
     *        step[i] = (initStep + n)^(-beta)  where n stand for  completed iterations。
     * lambda : The parameter of regularization
     * ratingFilePath: The original rating filePath
     * uFilePath: The User latent vector save filePath
     * vFilePath: The item latent vector save filePath
     *
     */
    /// 设置默认参数
    def this() = this(100,16,20,0.5,0.01,"hdfs://master:9000/user/chenalong/MovieLens/trainData",
        "hdfs://master:9000/user/chenalong/MovieLens/userLatentVector",
        "hdfs://master:9000/user/chenalong/MovieLens/itemLatentVector",10)
    

}

object DSGD
{

}
