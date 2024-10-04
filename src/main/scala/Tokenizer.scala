import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
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
      logger.info(s"Started running Mapper with key: $key")

      value.toString.toLowerCase().split("\\W+").filter(_.nonEmpty).foreach(token => {
        val encodedString = encoder.encode(token)
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
        throw new RuntimeException(s"Failed to encode token: $value", e)
    }
  }

  // TODO: remove?
  def decode(token: util.List[Integer]): String = {
    encoder.decode(token)
  }

  @main
  def tokenizerMain(inputPath: String, outputPath: String): RunningJob = {
    val conf: JobConf = new JobConf(this.getClass)
    conf.setJobName("Tokenizer")
    conf.set("fs.defaultFS", "hdfs://localhost:9000")
    conf.setLong("mapreduce.input.fileinputformat.split.maxsize", 6710) // 64 MB
    conf.setOutputKeyClass(classOf[Text])
    conf.setOutputValueClass(classOf[IntWritable])
    conf.setMapperClass(classOf[TokenizerMapper])
    conf.setCombinerClass(classOf[IntSumReducer])
    conf.setReducerClass(classOf[IntSumReducer])
    conf.setInputFormat(classOf[TextInputFormat])
    conf.setOutputFormat(classOf[TextOutputFormat[Text, IntWritable]])
    FileInputFormat.setInputPaths(conf, new Path(inputPath))
    FileOutputFormat.setOutputPath(conf, new Path(outputPath))

    logger.info("Starting the MapReduce job")
    val job = JobClient.runJob(conf)
    logger.info("Job completed successfully")
    job
  }

  @main
  def tokenizerTest(): Unit = {
    val tokenizer = Encodings.newDefaultEncodingRegistry().getEncoding(EncodingType.CL100K_BASE)

    val input = "Hello world! This is a sample program.\n We are going to check how close the Hello world is."
    val tokenIds = encode(input)
    val output = decode(tokenIds)

    println(s"Original Text: $input")
    println(s"Token IDs: $tokenIds")
    println(s"Original text back: $output")
  }
}
