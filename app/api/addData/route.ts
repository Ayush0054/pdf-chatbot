import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { NextRequest, NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { TaskType } from "@google/generative-ai";

import { PineconeStore } from "langchain/vectorstores/pinecone";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

export async function POST(request: NextRequest) {
  const data = await request.formData();

  const file: File | null = data.get("file") as unknown as File;

  if (!file) {
    return NextResponse.json({ success: false, error: "No file found" });
  }

  if (file.type !== "application/pdf") {
    return NextResponse.json({ success: false, error: "Invalid file type" });
  }

  const pdfLoader = new PDFLoader(file);
  const splitDocuments = await pdfLoader.loadAndSplit();

  const pineconeClient = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY ?? "",
    environment: "gcp-starter",
  });
  //@ts-ignore

  const pineconeIndex = pineconeClient.Index(
    process.env.PINECONE_INDEX_NAME as string
  );

  await PineconeStore.fromDocuments(
    splitDocuments,
    new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_API_KEY ?? "",
      // title: "One",
      modelName: "models/embedding-001",
      // taskType: TaskType.RETRIEVAL_DOCUMENT,
    }),
    {
      pineconeIndex,
    }
  );
  console.log("success", splitDocuments);

  return NextResponse.json({ success: true });
}
