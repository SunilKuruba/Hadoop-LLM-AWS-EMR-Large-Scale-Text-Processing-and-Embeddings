import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.*
import org.apache.hadoop.mapred.*

import java.io.IOException
import java.util
import scala.jdk.CollectionConverters.*

object Tokenizer {
  private val encoding = Encodings.newDefaultEncodingRegistry().getEncoding(EncodingType.CL100K_BASE)

  class TokenizerMapper extends MapReduceBase with Mapper[LongWritable, Text, Text, IntWritable] {
    private final val one = new IntWritable(1)
    private val word = new Text()

    @throws[IOException]
    override def map(key: LongWritable, value: Text, output: OutputCollector[Text, IntWritable], reporter: Reporter): Unit =
      val tokens = encode(value.toString)
      tokens.asScala.foreach { token =>
        val decodedWord = decode(token)
        word.set(decodedWord+"\t"+token)
        output.collect(word, one)
      }
  }

  class IntSumReducer extends MapReduceBase with Reducer[Text, IntWritable, Text, IntWritable] {
    override def reduce(key: Text, values: util.Iterator[IntWritable], output: OutputCollector[Text, IntWritable], reporter: Reporter): Unit =
      val sum = values.asScala.map(_.get()).sum;
      output.collect(key, new IntWritable(sum))
  }

  def encode(value: String): util.List[Integer] = {
    encoding.encode(value)
  }

  def decode(token: Integer): String = {
    encoding.decode(List(token).asJava)
  }

  @main
  def tokenizerMain(inputPath: String, outputPath: String): RunningJob = {
    val conf: JobConf = new JobConf(this.getClass)
    conf.setJobName("Tokenizer")
    conf.set("fs.defaultFS", "hdfs://localhost:9000")
    // Set the maximum split size
    //    conf.setLong("mapreduce.input.fileinputformat.split.maxsize", 6710) // 64 MB
    conf.setOutputKeyClass(classOf[Text])
    conf.setOutputValueClass(classOf[IntWritable])
    conf.setMapperClass(classOf[TokenizerMapper])
    conf.setCombinerClass(classOf[IntSumReducer])
    conf.setReducerClass(classOf[IntSumReducer])
    conf.setInputFormat(classOf[TextInputFormat])
    conf.setOutputFormat(classOf[TextOutputFormat[Text, IntWritable]])
    FileInputFormat.setInputPaths(conf, new Path(inputPath))
    FileOutputFormat.setOutputPath(conf, new Path(outputPath))
    JobClient.runJob(conf)
  }

  @main
  def tokenizerTest(): Unit = {
    val tokenizer = Encodings.newDefaultEncodingRegistry().getEncoding(EncodingType.CL100K_BASE)

    val input = "Hello world! This is a sample program.\n We are going to check how close the Hello world is."
    val tokenIds = tokenizer.encode(input)
    val output =  tokenizer.decode(tokenIds)

    println(s"Original Text: $input")
    println(s"Token IDs: $tokenIds")
    println(s"Original text back: $output")
  }
}
