"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useCompletion } from "ai/react";

export default function Home() {
  // When a file is dropped in the dropzone, call the `/api/addData` API to train our bot on a new PDF File
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];

    if (file.type !== "application/pdf") {
      alert("Please upload a PDF");
      return;
    }

    const formData = new FormData();
    formData.set("file", file);

    const response = await fetch("/api/addData", {
      method: "POST",
      body: formData,
    });
    console.log(response);

    const body = await response.json();
    console.log("body");

    console.log(body);

    if (body.success) {
      alert("Data added successfully");
    }
  }, []);

  // Configure react-dropzone
  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
  });

  // Vercel AI hook for generating completions through an AI model
  const { completion, input, isLoading, handleInputChange, handleSubmit } =
    useCompletion({
      api: "/api/chat",
    });
  console.log(completion);

  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <div
        {...getRootProps({
          className:
            "dropzone bg-blue-300 shadow-lg p-10 rounded-md hover:bg-blue-400 transition-colors duration-200 ease-in-out cursor-pointer",
        })}
      >
        <input {...getInputProps()} />
        <p>Upload a PDF to add new data</p>
      </div>

      <div className="mx-auto w-full items-center max-w-md py-24 flex flex-col stretch">
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <input
            className=" w-full max-w-md text-black border border-gray-300 rounded shadow-xl p-2"
            value={input}
            placeholder="Enter your prompt..."
            onChange={handleInputChange}
          />

          <button
            disabled={isLoading}
            type="submit"
            className="py-2 border shadow-md border-blue-300 text-blue-300 bg-white hover:text-white rounded-lg hover:bg-blue-300 text-sm px-6"
          >
            Submit
          </button>

          <p className="text-center">
            Completion result: {completion === "" ? "Thinking..." : completion}
          </p>
        </form>
      </div>
    </main>
  );
}
