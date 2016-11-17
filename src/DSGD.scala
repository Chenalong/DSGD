
import java.io.PrintWriter
import java.util.Date

import _root_.DSGD.DSGDParams
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD

import org.apache.spark.{SparkContext, SparkConf, Logging}

import scopt.OptionParser

import scala.util.Random

import scala.collection.Map

/**
 * Created by Administrator on 2016/6/15 0015.
 */
class DSGD(rank: Int, initStep: Float, numWorkers: Int, numIterations: Int, beta: Float, lambda: Float, ratingFilePath: String,
           uFilePath: String, vFilePath: String) extends Serializable with Logging
{
    /**
     * numFactors : The length of user or item hide feature vector
     * numWorkers: The degree of parallelism
     * numIterations: The quanlity of iterations
     * initStep : initial update steps
     * beta : The parameter of steps, following is update formula
     * step[i] = (initStep + n)^^(-beta)  where n stand for  completed iterations。
     * lambda : The parameter of regularization
     * ratingFilePath: The original rating filePath
     * uFilePath: The User latent vector save filePath
     * vFilePath: The item latent vector save filePath
     *
     */

    var seed = 0
    var updatesTotal = 0

    def printParameterValue(): Unit =
    {
        var outStr = "DSGD Parameters as following:\n"
        outStr += s"rank = ${rank}\n"
        outStr += s"initStep = ${initStep}\n"
        outStr += s"numWorkers = ${numWorkers}\n"
        outStr += s"numIterations = ${numIterations}\n"
        outStr += s"beta = ${beta}\n"
        outStr += s"lambda = ${lambda}\n"
        outStr += s"ratingFilePath = ${ratingFilePath}\n"
        outStr += s"uFilePath = ${uFilePath}\n"
        outStr += s"vFilePath = ${vFilePath}\n"
        println(outStr)
    }

    def train(ratings: RDD[Rating], params: DSGDParams, sc: SparkContext): Unit =
    {
        //        统计一部电影被多少用户观看，和一个用户看多少部电影
        val userRatingMovieNum = ratings.keyBy(rating => rating.user).countByKey()
        val movieRatedByUserNum = ratings.keyBy(rating => rating.item).countByKey()

        val userMaxId = ratings.map(_.user).distinct().reduce(math.max)
        val itemMaxId = ratings.map(_.item).distinct().reduce(math.max)

        println(s"userMaxId = $userMaxId and itemMaxId = $itemMaxId")
        //        定义 用户和电影统计数据输出文件名
        val userRatingMovieNumFilePath = params.dataName + "_userRatingMovieNum.data"
        val movieRatedByUserNumFilePath = params.dataName + "_movieRatedByUserNum.data"
        //      定义文件输出句柄
        val userRatingMovieNumOut = new PrintWriter(userRatingMovieNumFilePath)
        val movieRatedByUserNumOut = new PrintWriter(movieRatedByUserNumFilePath)
        //      数据输出到文件中
        userRatingMovieNumOut.println("The file record every user rated movies num")
        movieRatedByUserNumOut.println("The file record every movie rated by how many users")
        userRatingMovieNum.foreach(p => userRatingMovieNumOut.print(s"${p._1}\t${p._2}\n"))
        movieRatedByUserNum.foreach(p => movieRatedByUserNumOut.print(s"${p._1}\t${p._2}\n"))
        //      文件句柄关闭
        userRatingMovieNumOut.close()
        movieRatedByUserNumOut.close()


        var iter = 0


        var U = sc.parallelize(Range(0, userMaxId + 1)).map(ConstructLatentVector).persist()
        var V = sc.parallelize(Range(0, itemMaxId + 1)).map(ConstructLatentVector).persist()

        while (iter < params.numIterations)
        {
            seed = ObtainCurrentTimeStamp() % 1000000007

            var Ublocks = U.keyBy(v => RandomPartition(v._1))
            var Vblocks = V.keyBy(v => RandomPartition(v._1))

            val diagonalRatingData = ratings.filter(line => FilterDiagonal(line.user, line.item)).persist()

            val diagonalBlocks = diagonalRatingData.keyBy(v => RandomPartition(v.user))

            val updateSampleNum = diagonalBlocks.count()

            diagonalRatingData.unpersist()

            val ratingAndLatentRDD = diagonalBlocks.groupWith(Ublocks, Vblocks).coalesce(numWorkers)

            val updateUV = ratingAndLatentRDD.map{p=>UpdateWithSGD(p._1,p._2._1,p._2._2,p._2._3,userRatingMovieNum,
                movieRatedByUserNum)}.persist()

            U = updateUV.flatMap(p=> p._1)
            V = updateUV.flatMap(p=> p._2)
            updatesTotal += updateSampleNum
            iter += 1
        }

        WriteLatentToLocalFile(U,V,params.dataName)

//        val userLatendVector = U.sortByKey().map(p=>p._2).collect()
//        val itemLatendVector = V.sortByKey().map(p=>p._2).collect()
    }

    def UpdateWithSGD(Id: Int, ratingBlock: scala.Iterable[Rating], Ublock: scala.Iterable[(Int, Array[Float])],
                      Vblock: scala.Iterable[(Int, Array[Float])], userRatingMovieNum: Map[Int, Long],
                      movieRatedByUserNum: Map[Int, Long]): (scala.Iterable[(Int, Array[Float])],
        scala.Iterable[(Int, Array[Float])]) =
    {
        val uDict = Ublock.toMap
        val vDict = Vblock.toMap

        var iter = 0


        ratingBlock.map
        { p =>
            iter += 1
            val userId = p.user
            val itemId = p.item
            val score = p.score

            val epsion = math.pow(100 + updatesTotal + iter, -1 * beta).toFloat

            val ui = uDict(userId)
            val vi = vDict(itemId)
            val predictScore = ui.view.zip(vi.view).map(p => p._1 * p._2).sum
            var loss = -2 * (score - predictScore)

            for (i <- 0 to rank)
            {
                uDict(userId)(i) = ui(i) - epsion * (2 * lambda / userRatingMovieNum(userId) * ui(i) + loss * vi(i))
                vDict(itemId)(i) = vi(i) - epsion * (2 * lambda / movieRatedByUserNum(itemId) * vi(i) + loss * ui(i))
            }
            p
        }

        (uDict.toList, vDict.toList)
    }

    def FilterDiagonal(a: Int, b: Int): Boolean =
    {
        RandomPartition(a) == RandomPartition(b)
    }

    def RandomPartition(Id: Int): Int =
    {
        val b = Id.toString() + seed.toString
        b.hashCode % numWorkers
    }


    def ConstructLatentVector(Id: Int): (Int, Array[Float]) =
    {
        val r = new Random(ObtainCurrentTimeStamp)
        val factor = Array.fill(rank)(r.nextFloat())
        (Id, factor)
    }

    def ObtainCurrentTimeStamp(): Int =
    {
        val now = new Date()
        var timeStamp = now.getTime + ""
        timeStamp.substring(0, 10).toInt
    }

    def WriteLatentToLocalFile(U:RDD[(Int,Array[Float])],V:RDD[(Int,Array[Float])],dataName:String):Unit = {
        val userLatendVector = U.sortByKey().collect()
        val itemLatendVector = V.sortByKey().collect()

        val userLatentFilePath = dataName + "UserLatent.data"
        val itemLatentFilePath = dataName + "ItemLatent.data"
        val userLatentOut = new PrintWriter(userLatentFilePath)
        val itemLatentOut = new PrintWriter(itemLatentFilePath)
        //      数据输出到文件中
        userLatentOut.println("The file record users Latent vector")
        itemLatentOut.println("The file record items Latent vector")

        for(i<-0 until userLatendVector.length)
        {
            userLatentOut.print(s"${userLatendVector(i)._1}: ")
            userLatendVector(i)._2.foreach(p=> userLatentOut.print(s"${p} "))
            userLatentOut.print("\n")
        }

        for(i<-0 until itemLatendVector.length)
        {
            itemLatentOut.print(s"${itemLatendVector(i)._1}: ")
            itemLatendVector(i)._2.foreach(p=> itemLatentOut.print(s"${p} "))
            itemLatentOut.print("\n")
        }

        //      文件句柄关闭
        userLatentOut.close()
        itemLatentOut.close()

    }
}

object DSGD
{
    def main(args: Array[String]): Unit =
    {
        val defaultDSGDParams = DSGDParams()
        //        对输入参数的解析
        val parser = new OptionParser[DSGDParams]("DSGD algorithm parameter")
        {
            head("The DSGD algorithem parameters list")

            opt[Int]("rank")
                .text(s"rank, default: ${defaultDSGDParams.rank}")
                .action((x, c) => c.copy(rank = x))

            opt[Int]("numWorkers")
                .text(s"numWorkers, default: ${defaultDSGDParams.numWorkers}")
                .action((x, c) => c.copy(numWorkers = x))

            opt[Int]("numIterations")
                .text(s"numIterations, default: ${defaultDSGDParams.numIterations}")
                .action((x, c) => c.copy(numIterations = x))

            opt[Float]("beta")
                .text(s"beta, default: ${defaultDSGDParams.beta}")
                .action((x, c) => c.copy(beta = x))

            opt[Float]("lambda")
                .text(s"lambda, default: ${defaultDSGDParams.lambda}")
                .action((x, c) => c.copy(lambda = x))

            opt[Float]("initStep")
                .text(s"initStep, default: ${defaultDSGDParams.initStep}")
                .action((x, c) => c.copy(initStep = x))

            opt[String]("dataName")
                .text(s"dataName, default: ${defaultDSGDParams.dataName}")
                .action((x, c) => c.copy(dataName = x))

            opt[String]("ratingFilePath")
                .required()
                .text(s"original rating data filePath")
                .action((x, c) => c.copy(ratingFilePath = x))

            opt[String]("uFilePath")
                .required()
                .text(s"user latent vector save file path")
                .action((x, c) => c.copy(uFilePath = x))

            opt[String]("vFilePath")
                .required()
                .text(s"item latent vector save file path")
                .action((x, c) => c.copy(vFilePath = x))

            note(
                """
                  |For example, the following command runs this app on a synthetic dataset:
                  |
                  |bin/spark-submit --class DSGD \
                  |--jar examples/target/scala-*/spark-examples-*.jar \
                  |--rank 5 --numIterations 20 --lambda 1.0  \
                  |originalRatingFilePath uLatentVectorFilePath vLatentVectorFilePath
                """.stripMargin)
        }

        parser.parse(args, defaultDSGDParams).map
        { params => run(params)

        } getOrElse
            {
                println("===============Parameters is wrong=====================")
                System.exit(1)

            }

    }


    def run(params: DSGDParams)
    {
        val conf = new SparkConf().setAppName(params.dataName + s" DSGD with $params").setMaster("spark://master:7077")

        val sc = new SparkContext(conf)

        // sc.setCheckpointDir("hdfs://master:9000/user/chenalong/checkPointDir/")

        Logger.getRootLogger.setLevel(Level.WARN)

        // 从HDFS文件系统中读取数据，形成RDD
        val ratings = sc.textFile(params.ratingFilePath).map
        { line =>
            val fields = line.split('\t')
            Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat)
        }.cache()

        val DSGDExample = new DSGD(params.rank,params.initStep,params.numWorkers,params.numIterations,params.beta,
            params.lambda,params.ratingFilePath,params.uFilePath,params.vFilePath)

        DSGDExample.train(ratings,params,sc)


        /*
        println(s"Got $numRatings ratings from $numUsers users on $numMovies movies.")

        val splits = ratings.randomSplit(Array(0.8, 0.2))
        val training = splits(0).cache()
        val test = splits(1).cache()

        val numTraining = training.count()
        val numTest = test.count()
        println(s"Training: $numTraining, test: $numTest.")

        //清理内存空间
        ratings.unpersist(blocking = false)



        val rmse = computeRmse(model, test, params.implicitPrefs)

        println(s"Test RMSE = $rmse.")
        */
        sc.stop()
    }

    case class DSGDParams(
                             rank: Int = 100,
                             numWorkers: Int = 16,
                             numIterations: Int = 20,
                             beta: Float = 0.5,
                             lambda: Float = 0.02,
                             initStep: Float = 10,
                             dataName: String = "RatingData",
                             ratingFilePath: String = "hdfs://master:9000/user/chenalong/MovieLens/trainData",
                             uFilePath: String = "hdfs://master:9000/user/chenalong/MovieLens/userLatentVector",
                             vFilePath: String = "hdfs://master:9000/user/chenalong/MovieLens/itemLatentVector"
                             ) extends MyAbstractParams[DSGDParams]

}
