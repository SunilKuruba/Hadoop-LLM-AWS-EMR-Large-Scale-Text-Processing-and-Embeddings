import JobConfig.Environment.test
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapred.{OutputCollector, Reporter}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EmbeddingSpec extends AnyFlatSpec with Matchers {
  JobConfig.environment = test

  // Stub implementation of OutputCollector
  class TestOutputCollector extends OutputCollector[Text, Text] {
    var collectedData: Map[Text, Text] = Map.empty

    override def collect(key: Text, value: Text): Unit = {
      collectedData += (key -> value)
    }
  }

  // Stub implementation of Iterator
  class TestIterator(values: Seq[String]) extends java.util.Iterator[Text] {
    private var index = 0

    override def hasNext: Boolean = index < values.length

    override def next(): Text = {
      val nextValue = new Text(values(index))
      index += 1
      nextValue
    }
  }

  "EmbeddingMapper" should "correctly tokenize input and emit token embeddings" in {
    val mapper = new Embedding.EmbeddingMapper
    val outputCollector = new TestOutputCollector()
    val reporter = Reporter.NULL

    // Arrange Simulated input
    val inputKey = new LongWritable(1)
    val inputValue = new Text("This is a test sentence")

    // Act
    mapper.map(inputKey, inputValue, outputCollector, reporter)

    // Assert
    outputCollector.collectedData should have size 4
    outputCollector.collectedData.foreach { case (key, value) =>
      key.toString should include("\t") // Token and token id
      value.toString should include("[") // Embedding as an INDArray representation
    }
  }

  "EmbeddingReducer" should "correctly calculate the average embedding for a token" in {
    val reducer = new Embedding.EmbeddingReducer
    val outputCollector = new TestOutputCollector()
    val reporter =  Reporter.NULL

    // Arrange
    val inputKey = new Text("test")
    val embeddings = new TestIterator(Seq("[1.0, 2.0]", "[2.0, 3.0]"))

    // Act
    reducer.reduce(inputKey, embeddings, outputCollector, reporter)

    // Assert
    outputCollector.collectedData should have size 1
    val (key, value) = outputCollector.collectedData.head
    key.toString shouldBe "test"
    value.toString shouldBe "[1.5, 2.5]" // Average of [1.0, 2.0] and [2.0, 3.0]
  }

  "EmbeddingMapper" should "not emit any tokens or embeddings when input is empty" in {
    val mapper = new Embedding.EmbeddingMapper
    val outputCollector = new TestOutputCollector()
    val reporter = Reporter.NULL

    // Empty input
    val inputKey = new LongWritable(1)
    val inputValue = new Text("")

    // Call map function
    mapper.map(inputKey, inputValue, outputCollector, reporter)

    // Verify that no tokens were emitted
    outputCollector.collectedData shouldBe empty
  }

  "EmbeddingMapper" should "correctly process a large input and emit multiple tokens and embeddings" in {
    val mapper = new Embedding.EmbeddingMapper
    val outputCollector = new TestOutputCollector()
    val reporter = Reporter.NULL

    // Simulate a large text input
    val inputKey = new LongWritable(1)
    val inputValue = new Text("The quick brown fox jumps over the lazy dog. " * 100)

    // Call map function
    mapper.map(inputKey, inputValue, outputCollector, reporter)

    // Verify that the mapper emitted tokens
    outputCollector.collectedData should not be empty
  }

  "EmbeddingMain" should "successfully configure and run the MapReduce job" in {
    val job = Embedding.embeddingMain()

    // Validate that the job completed successfully
    job.isComplete shouldBe true
    job.isSuccessful shouldBe true
  }
}
