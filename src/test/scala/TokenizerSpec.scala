import JobConfig.Environment.test
import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapred.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.util
import scala.collection.mutable
import scala.jdk.CollectionConverters.*

class TokenizerSpec extends AnyFlatSpec with Matchers {
  JobConfig.environment = test

  "Tokenizer" should "encode a string successfully" in {
    val input = "hello"
    val encoded = Tokenizer.encode(input)

    val expected = Tokenizer.decode(encoded)
    input shouldBe expected
  }

  "Tokenizer" should "decode a single encoded integer successfully" in {
    val encoded = 15339 // This is the actual encoded integer for "hello"
    val decoded = Tokenizer.decode(encoded)

    decoded shouldBe "hello"
  }

  "Tokenizer" should "decode a list of encoded integers successfully" in {
    val encoded = new util.ArrayList[Integer]()
    encoded.add(15339)
    encoded.add(1917)

    val decoded = Tokenizer.decode(encoded)

    decoded shouldBe "hello world"
  }

  "Tokenizer" should "handle decoding exceptions" in {
    val invalidEncodedValue = -1
    val thrown = intercept[Exception] {
      Tokenizer.decode(invalidEncodedValue)
    }
    thrown.getMessage should include("Failed to decode token")
  }

  "Tokenizer" should "return an empty list for encoding an empty string" in {
    val input = ""
    val encoded = Tokenizer.encode(input)

    encoded shouldBe empty
  }

  "Tokenizer" should "return an empty string for decoding an empty list" in {
    val emptyList = new util.ArrayList[Integer]()
    val decoded = Tokenizer.decode(emptyList)

    decoded shouldBe ""
  }

  "TokenizerMapper" should "handle empty input without errors" in {
    val key = new LongWritable(1)
    val value = new Text("")
    val output = new mutable.HashMap[Text, IntWritable]()

    val mapper = new Tokenizer.TokenizerMapper()
    mapper.map(key, value, (k, v) => output.put(k, v), null)

    output shouldBe empty
  }

  "IntSumReducer" should "sum multiple values for the same key" in {
    val key = new Text("hello")
    val values = List(new IntWritable(1), new IntWritable(2), new IntWritable(3)).asJava
    val output = new mutable.HashMap[Text, IntWritable]()

    val reducer = new Tokenizer.IntSumReducer()
    reducer.reduce(key, values.iterator(), (k, v) => output.put(k, v), null)

    output(new Text("hello")).toString shouldBe "6"
  }

  "Tokenizer MapReduce chain" should "handle multiple jobs correctly" in {
    // Input for first job
    val inputText = List(
      new LongWritable(1) -> new Text("first job input")
    )

    // Run the first job
    val firstJobOutput = new mutable.HashMap[Text, IntWritable]()
    val tokenizerMapper = new Tokenizer.TokenizerMapper()
    inputText.foreach { case (key, value) =>
      tokenizerMapper.map(key, value, (k, v) => firstJobOutput.put(k, v), null)
    }

    // Use first job's output as the second job's input
    val secondJobInput = firstJobOutput.toList.map {
      case (text, count) => new LongWritable(1) -> new Text(text.toString)
    }

    val secondJobOutput = new mutable.HashMap[Text, IntWritable]()
    secondJobInput.foreach { case (key, value) =>
      tokenizerMapper.map(key, value, (k, v) => secondJobOutput.put(k, v), null)
    }

    // Validate the final output
    secondJobOutput should not be empty
  }

  "Tokenizer MapReduce job" should "handle empty input file without errors" in {
    // Simulate an empty input file
    val inputText = List.empty[(LongWritable, Text)]

    // Run the Mapper
    val mapperOutput = new mutable.HashMap[Text, IntWritable]()
    val tokenizerMapper = new Tokenizer.TokenizerMapper()

    inputText.foreach { case (key, value) =>
      tokenizerMapper.map(key, value, (k, v) => mapperOutput.put(k, v), null)
    }

    // Run the Reducer
    val reducerOutput = new mutable.HashMap[Text, IntWritable]()
    val tokenizerReducer = new Tokenizer.IntSumReducer()

    mapperOutput.groupBy(_._1).foreach { case (key, values) =>
      val iter = values.map(_._2).iterator.asJava
      tokenizerReducer.reduce(key, iter, (k, v) => reducerOutput.put(k, v), null)
    }

    // The output should be empty
    reducerOutput shouldBe empty
  }

  "Tokenizer MapReduce job" should "produce correct output for a sample input" in {
    val inputText = List(
      new LongWritable(1) -> new Text("hello world"),
      new LongWritable(2) -> new Text("foo bar")
    )

    val expectedOutput = Map(
      new Text("hello\t[15339]") -> new IntWritable(1),
      new Text("world\t[14957]") -> new IntWritable(1),
      new Text("foo\t[8134]") -> new IntWritable(1),
      new Text("bar\t[2308]") -> new IntWritable(1)
    )

    val mapperOutput = new mutable.HashMap[Text, IntWritable]()

    // Run the Mapper
    val tokenizerMapper = new Tokenizer.TokenizerMapper()
    inputText.foreach { case (key, value) =>
      tokenizerMapper.map(key, value, (k, v) => mapperOutput.put(new Text(k), new IntWritable(v.get())), null)
    }

    // Run the Reducer with the Mapper's output
    val reducerOutput = new mutable.HashMap[Text, IntWritable]()
    val tokenizerReducer = new Tokenizer.IntSumReducer()

    mapperOutput.groupBy(_._1).foreach { case (key, values) =>
      val iter = values.map(_._2).iterator.asJava // This converts the Scala iterator to a Java one
      tokenizerReducer.reduce(key, iter, (k, v) => reducerOutput.put(new Text(k), new IntWritable(v.get())), null)
    }

    // Validate output
    reducerOutput shouldBe expectedOutput
  }

  "Tokenizer MapReduce job" should "run e2e locally" in {
    val random = Math.random()
    val job = Tokenizer.tokenizerMain()

    // Validate that the job completed successfully
    job.isComplete shouldBe true
    job.isSuccessful shouldBe true
  }
}