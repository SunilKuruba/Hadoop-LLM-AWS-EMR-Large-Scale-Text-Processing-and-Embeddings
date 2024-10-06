import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
import org.apache.hadoop.io.*
import org.apache.hadoop.mapred.*
import org.slf4j.{Logger, LoggerFactory}

import java.io.IOException
import java.util
import scala.jdk.CollectionConverters.*

/**
 * The Tokenizer object is responsible for tokenizing text input using Hadoop's MapReduce framework.
 * It encodes tokens and processes text files to compute token frequencies using a Mapper and Reducer.
 */
object Tokenizer {

  /**
   * Encoder used to encode and decode tokens based on a specific encoding type.
   */
  private val encoder = Encodings.newDefaultEncodingRegistry().getEncoding(EncodingType.CL100K_BASE)

  /**
   * Logger for logging important events and errors during the tokenization process.
   */
  private val logger: Logger = LoggerFactory.getLogger(Tokenizer.getClass)

  /**
   * TokenizerMapper is a MapReduce Mapper class. It tokenizes input text, encodes each token,
   * and emits the token and its encoded value as the key, with a value of 1.
   */
  class TokenizerMapper extends MapReduceBase with Mapper[LongWritable, Text, Text, IntWritable] {
    private final val one = new IntWritable(1)
    private val outputKey = new Text()

    /**
     * Map function to process each input line. It splits the text into tokens, encodes each token,
     * and collects the (token, encoded value) pair as key, and 1 as the value.
     *
     * @param key      The line number of the input split.
     * @param value    The line content of the input split.
     * @param output   The output collector to store the intermediate (key, value) pairs.
     * @param reporter Reporter to handle progress and status updates.
     * @throws IOException If any I/O error occurs during processing.
     */
    @throws[IOException]
    override def map(key: LongWritable, value: Text, output: OutputCollector[Text, IntWritable], reporter: Reporter): Unit =
      logger.info(s"Started running Tokenizer Mapper with key: $key")

      value.toString.toLowerCase().split("\\W+").filter(_.nonEmpty).foreach(token => {
        val encodedString = encode(token)
        outputKey.set(token + "\t" + encodedString)
        output.collect(outputKey, one)
      })
  }

  /**
   * IntSumReducer is a MapReduce Reducer class. It aggregates the counts of tokens by summing up the
   * values (which are all 1s emitted by the Mapper) for each token.
   */
  class IntSumReducer extends MapReduceBase with Reducer[Text, IntWritable, Text, IntWritable] {

    /**
     * Reduce function to sum up the values for each token and output the total count.
     *
     * @param key      The token and its encoded value.
     * @param values   The iterator over all the values associated with this key.
     * @param output   The output collector to store the final (key, value) pair.
     * @param reporter Reporter to handle progress and status updates.
     */
    override def reduce(key: Text, values: util.Iterator[IntWritable], output: OutputCollector[Text, IntWritable], reporter: Reporter): Unit =
      val sum = values.asScala.map(_.get()).sum
      output.collect(key, new IntWritable(sum))
  }

  /**
   * Encodes the given string value into a list of integers using the predefined encoder.
   *
   * @param value The string to encode.
   * @return A list of integers representing the encoded string.
   * @throws RuntimeException If encoding fails.
   */
  def encode(value: String): util.List[Integer] = {
    try {
      encoder.encode(value)
    } catch {
      case e: Exception =>
        throw new RuntimeException(s"Failed to encode token: $value", e)
    }
  }

  /**
   * Decodes a given integer back into its string representation using the predefined encoder.
   *
   * @param value The integer to decode.
   * @return The decoded string.
   * @throws RuntimeException If decoding fails.
   */
  def decode(value: Integer): String = {
    try {
      encoder.decode(List(value).asJava)
    } catch {
      case e: Exception =>
        throw new RuntimeException(s"Failed to decode token: $value", e)
    }
  }

  /**
   * Decodes a list of integers back into their string representation using the predefined encoder.
   *
   * @param token The list of integers to decode.
   * @return The decoded string.
   */
  def decode(token: util.List[Integer]): String = {
    encoder.decode(token)
  }

  /**
   * The main entry point for the MapReduce job. It sets up the job configuration,
   * including the Mapper and Reducer classes, and starts the MapReduce job.
   *
   * @return The RunningJob object representing the MapReduce job.
   */
  @main
  def tokenizerMain(): RunningJob = {
    val jobConf: JobConf = JobConfig.createJob("hadoop.tokenizer.jobName")
    jobConf.setOutputKeyClass(classOf[Text])
    jobConf.setOutputValueClass(classOf[IntWritable])
    jobConf.setMapperClass(classOf[TokenizerMapper])
    jobConf.setCombinerClass(classOf[IntSumReducer])
    jobConf.setReducerClass(classOf[IntSumReducer])
    jobConf.setInputFormat(classOf[TextInputFormat])
    jobConf.setOutputFormat(classOf[TextOutputFormat[Text, IntWritable]])

    logger.info("Starting the MapReduce job")
    val job = JobClient.runJob(jobConf)
    logger.info("Job completed successfully")
    job
  }
}
