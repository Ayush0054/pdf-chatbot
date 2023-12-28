import { NextRequest } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { TaskType } from "@google/generative-ai";

import { loadQAChain } from "langchain/chains";
import { StreamingTextResponse, LangChainStream } from "ai";

import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { PromptTemplate } from "langchain/prompts";
import { CallbackManager } from "langchain/callbacks";

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
      modelName: "models/embedding-001",
      // taskType: TaskType.RETRIEVAL_DOCUMENT,
    }),
    //@ts-ignore
    { pineconeIndex }
  );
  const vectorStoreRetrieval = vectorStore.asRetriever();
  const docs = await vectorStoreRetrieval.getRelevantDocuments(body.prompt);
  console.log("docs", docs);

  const model = new ChatGoogleGenerativeAI({
    apiKey: process.env.GOOGLE_API_KEY ?? "",
    temperature: 0.7,
    modelName: "gemini-pro",
    // topK: 40,
    // topP: 1,
    // maxOutputTokens: 2048,
    callbackManager: CallbackManager.fromHandlers(handlers),
  });
  const promptTemplate = `
  Please answer the question in as much detail as possible based on the provided context.
  Ensure to include all relevant details.

  Context:
  {context}?

  Question:
  {question}

  Answer:
`;
  const prompt = new PromptTemplate({
    template: promptTemplate,
    inputVariables: ["context", "question"],
  });

  //@ts-ignore
  const chain = loadQAChain(model, {
    // k: 1,
    type: "stuff",
    prompt: prompt,
  });

  // console.log("chain-----------------------:", chain);

  // console.log("Executing chain with prompt:", body.prompt);

  //@ts-ignore
  const result = await chain.call({
    input_documents: docs,
    query: body.prompt,
    question: `${body.prompt}}`,
  });

  // console.log("Chain execution result:", result);
  // console.log(stream);

  //@ts-ignore
  return new StreamingTextResponse(stream);
}
