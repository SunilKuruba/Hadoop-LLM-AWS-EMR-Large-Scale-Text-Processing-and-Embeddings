val scala3Version = "3.5.0"

lazy val root = project
  .in(file("."))
  .settings(
    name := "LLM-hw1",
    version := "0.1.0-SNAPSHOT",
    scalaVersion := scala3Version,
    libraryDependencies ++= Seq(
      "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-beta7",      // Word2Vec dependency
      "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta7",              // ND4J library for vector operations
      "org.slf4j" % "slf4j-api" % "1.7.30",                            // SLF4J for logging
      "org.slf4j" % "slf4j-simple" % "1.7.30",                          // Simple SLF4J binding
      "org.apache.hadoop" % "hadoop-common" % "3.4.0",
      "org.apache.hadoop" % "hadoop-mapreduce-client-core" % "3.4.0",
      "org.apache.hadoop" % "hadoop-mapreduce-client-jobclient" % "3.4.0",
      "com.knuddels" % "jtokkit" % "0.6.1",
      "org.scalameta" % "munit" % "1.0.0" % Test,
      "org.scalatest" % "scalatest" % "3.2.9" % Test,
      "org.mockito" % "mockito-scala" % "1.16.46" % Test
    ),
    assemblyMergeStrategy in assembly := {
      case PathList("META-INF", xs @ _*) => MergeStrategy.discard
      case x => MergeStrategy.first
    }
  )