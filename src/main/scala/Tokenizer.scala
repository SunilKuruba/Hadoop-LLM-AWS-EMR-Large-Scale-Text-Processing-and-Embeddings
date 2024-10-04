import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
import com.typesafe.config.ConfigFactory
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.*
import org.apache.hadoop.mapred.*
import org.slf4j.{Logger, LoggerFactory}

import java.io.IOException
import java.util
import scala.jdk.CollectionConverters.*

object Tokenizer {
  private val encoder = Encodings.newDefaultEncodingRegistry().getEncoding(EncodingType.CL100K_BASE)
  private val logger: Logger = LoggerFactory.getLogger(Tokenizer.getClass)

  class TokenizerMapper extends MapReduceBase with Mapper[LongWritable, Text, Text, IntWritable] {
    private final val one = new IntWritable(1)
    private val outputKey = new Text()

    @throws[IOException]
    override def map(key: LongWritable, value: Text, output: OutputCollector[Text, IntWritable], reporter: Reporter): Unit =
      logger.info(s"Started running Tokenizer Mapper with key: $key")

      value.toString.toLowerCase().split("\\W+").filter(_.nonEmpty).foreach(token => {
        val encodedString = encode(token)
        outputKey.set(token + "\t" + encodedString)
        output.collect(outputKey, one)
      })
  }

  class IntSumReducer extends MapReduceBase with Reducer[Text, IntWritable, Text, IntWritable] {
    override def reduce(key: Text, values: util.Iterator[IntWritable], output: OutputCollector[Text, IntWritable], reporter: Reporter): Unit =
      val sum = values.asScala.map(_.get()).sum;
      output.collect(key, new IntWritable(sum))
  }

  def encode(value: String): util.List[Integer] = {
    try {
      encoder.encode(value)
    }catch {
      case e: Exception =>
        throw new RuntimeException(s"Failed to encode token: $value", e)
    }
  }

  def decode(value: Integer): String = {
    try {
      encoder.decode(List(value).asJava)
    } catch {
      case e: Exception =>
        throw new RuntimeException(s"Failed to decode token: $value", e)
    }
  }

  def decode(token: util.List[Integer]): String = {
    encoder.decode(token)
  }

  @main
  def tokenizerMain(inputPath: String, outputPath: String): RunningJob = {
    val config = ConfigFactory.load
    val jobConf: JobConf = new JobConf(this.getClass)
    jobConf.setJobName(config.getString("hadoop.tokenizer.jobName"))
    jobConf.set("fs.defaultFS",config.getString("hadoop.fs.defaultFS"))
    jobConf.setLong("mapreduce.input.fileinputformat.split.maxsize", config.getLong("hadoop.blockSize"))
    jobConf.setOutputKeyClass(classOf[Text])
    jobConf.setOutputValueClass(classOf[IntWritable])
    jobConf.setMapperClass(classOf[TokenizerMapper])
    jobConf.setCombinerClass(classOf[IntSumReducer])
    jobConf.setReducerClass(classOf[IntSumReducer])
    jobConf.setInputFormat(classOf[TextInputFormat])
    jobConf.setOutputFormat(classOf[TextOutputFormat[Text, IntWritable]])
    FileInputFormat.setInputPaths(jobConf, new Path(inputPath))
    FileOutputFormat.setOutputPath(jobConf, new Path(outputPath))

    logger.info("Starting the MapReduce job")
    val job = JobClient.runJob(jobConf)
    logger.info("Job completed successfully")
    job
  }
}
