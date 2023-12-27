import { NextRequest } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { TaskType } from "@google/generative-ai";

import {
  RetrievalQAChain,
  VectorDBQAChain,
  loadQAMapReduceChain,
} from "langchain/chains";
import { StreamingTextResponse, LangChainStream } from "ai";

import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";

export async function POST(request: NextRequest) {
  const body = await request.json();

  const { stream, handlers } = LangChainStream();

  const pineconeClient = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY ?? "",
    environment: "gcp-starter",
  });

  const pineconeIndex = pineconeClient.Index(
    process.env.PINECONE_INDEX_NAME as string
  );

  const vectorStore = await PineconeStore.fromExistingIndex(
    new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_API_KEY ?? "",
      // title: "One",
      modelName: "embedding-001",
      taskType: TaskType.RETRIEVAL_DOCUMENT,
    }),
    //@ts-ignore
    { pineconeIndex }
  );
  const results = await vectorStore.similaritySearch("java", 1, {
    blobType: "application/pdf",
  });
  console.log(results);
  // console.log(pineconeIndex);

  // console.log("vectorStore", vectorStore);

  const model = new ChatGoogleGenerativeAI({
    apiKey: process.env.GOOGLE_API_KEY ?? "",
    temperature: 0.7,
    modelName: "gemini-pro",
    topK: 40,
    topP: 1,
    maxOutputTokens: 2048,
    // callbackManager: CallbackManager.fromHandlers(handlers),
  });
  // Define the Langchain chain
  //@ts-ignore
  const chain = VectorDBQAChain.fromLLM(model, vectorStore, {
    // k: 1,
    returnSourceDocuments: true,
  });

  console.log("chain-----------------------:", chain);

  // Inside the POST function
  console.log("Executing chain with prompt:", body.prompt);

  // Call our chain with the prompt given by the user
  const result = await chain.call({ query: body.prompt }).catch(console.error);
  console.log("Chain execution result:", result);
  console.log(stream);

  // Return an output stream to the frontend
  //@ts-ignore
  return new StreamingTextResponse(result.text);
}
